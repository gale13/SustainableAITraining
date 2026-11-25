"""Build the final PV-only database from the already hash-verified official CSVs."""
import argparse,csv,hashlib,json,math,sqlite3,sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from opencem_pv import PVDatabase,PVDay

SCHEMA='''
CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE sources(id INTEGER PRIMARY KEY,path TEXT UNIQUE,sha256 TEXT,git_blob_sha1 TEXT,bytes INTEGER);
CREATE TABLE pv_observations(
 read_ts INTEGER PRIMARY KEY,inverter INTEGER NOT NULL CHECK(inverter=1),
 pv_power_raw_w REAL,raw_power_text TEXT,pv_power_w REAL,pv1volt REAL,pv1curr REAL,
 quality TEXT NOT NULL,source_id INTEGER NOT NULL,source_row INTEGER NOT NULL,
 row_hash BLOB NOT NULL,duplicate_count INTEGER NOT NULL DEFAULT 1);
CREATE INDEX valid_power_timestamp ON pv_observations(read_ts) WHERE pv_power_w IS NOT NULL;
CREATE VIEW analog_measurements AS SELECT read_ts,inverter,pv_power_w AS pv1power,pv1volt,pv1curr FROM pv_observations;
CREATE TABLE days(date TEXT PRIMARY KEY,raw_rows INTEGER,valid_rows INTEGER,invalid_rows INTEGER,
 first_observation_s REAL,last_observation_s REAL,median_interval_s REAL,max_support_gap_s REAL,
 coverage_60_s REAL,coverage_300_s REAL,pv_peak_w REAL,known_energy_300_j REAL,full_day_energy_j REAL,
 recommended INTEGER DEFAULT 0);
'''

def finite(s):
    if not s.strip():return None
    n=float(s)
    return n if math.isfinite(n) else None

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit',type=Path,default=ROOT.parent/'opencem-audit-20261123')
    parser.add_argument('--rebuild',action='store_true');args=parser.parse_args()
    target=ROOT/'src/opencem_pv/data/opencem_pv.sqlite';target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and not args.rebuild:raise FileExistsError('Database exists; use --rebuild explicitly')
    tmp=target.with_suffix('.building.sqlite')
    if tmp.exists():tmp.unlink()
    con=sqlite3.connect(tmp);con.executescript(SCHEMA)
    con.execute('INSERT INTO metadata VALUES (?,?)',('format_version','1'))
    con.execute('INSERT INTO metadata VALUES (?,?)',('timezone','Asia/Shanghai'))
    receipt=json.loads((args.audit/'download_receipt.json').read_text())
    sources=[x for x in receipt['files'] if x['kind']=='measurements']
    counts=Counter();per_source=[]
    for sid,source in enumerate(sources,1):
        path=args.audit/'raw'/source['path']
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024**2),b''):digest.update(chunk)
        assert digest.hexdigest()==source['sha256'],source['path']
        con.execute('INSERT INTO sources VALUES (?,?,?,?,?)',(sid,source['path'],source['sha256'],source['git_blob_sha1'],source['size']))
        batch=[];filecount=Counter()
        def flush():
            if not batch:return
            cur=con.executemany('''INSERT INTO pv_observations
                (read_ts,inverter,pv_power_raw_w,raw_power_text,pv_power_w,pv1volt,pv1curr,quality,source_id,source_row,row_hash)
                VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(read_ts) DO UPDATE SET duplicate_count=duplicate_count+1
                WHERE pv_observations.row_hash=excluded.row_hash''',batch)
            if cur.rowcount!=len(batch):raise ValueError('Conflicting records at the same timestamp; refusing to choose silently')
            batch.clear()
        with path.open(newline='') as f:
            reader=csv.reader(f);header=next(reader)
            short=[c for c in header if c not in ['battreserve','pvreserve','otherreserve','grid_reserve']]
            assert len(short)==78
            ti,ii,pi,vi,ci=[short.index(x) for x in ['read_ts','inverter','pv1power','pv1volt','pv1curr']]
            for number,row in enumerate(reader,2):
                filecount['all_rows']+=1
                if len(row)!=78:
                    assert len(row)==82
                    filecount['old_82_rows_skipped']+=1;continue
                if int(row[ii])!=1:
                    filecount['inverter_2_rows_skipped']+=1;continue
                filecount['selected_78_inverter_1_rows']+=1
                power=finite(row[pi]);quality='observed'
                if power is None:quality='missing' if not row[pi].strip() else 'nonfinite'
                elif power<0:quality='negative'
                cleaned=power if quality=='observed' else None
                batch.append((int(row[ti]),1,power,row[pi],cleaned,finite(row[vi]),finite(row[ci]),quality,sid,number,
                              hashlib.blake2b('\0'.join(row).encode(),digest_size=16).digest()))
                if len(batch)>=20000:flush()
            flush()
        con.commit();counts.update(filecount);per_source.append(dict(path=source['path'],**filecount))
        print('Imported',source['path'],filecount['selected_78_inverter_1_rows'],flush=True)
    unique=con.execute('SELECT COUNT(*) FROM pv_observations').fetchone()[0]
    assert unique==1135611,unique
    counts['unique_rows']=unique
    counts['exact_duplicate_rows_removed']=counts['selected_78_inverter_1_rows']-unique
    assert con.execute('SELECT SUM(duplicate_count) FROM pv_observations').fetchone()[0]==counts['selected_78_inverter_1_rows']
    quality=dict(con.execute('SELECT quality,COUNT(*) FROM pv_observations GROUP BY quality'))
    days=[r[0] for r in con.execute("SELECT DISTINCT date(read_ts,'unixepoch','+8 hours') FROM pv_observations ORDER BY read_ts")]
    assert len(days)==242,len(days)
    con.executemany('INSERT INTO days(date) VALUES (?)',[(d,) for d in days]);con.commit()
    with PVDatabase(tmp) as db:
        for date in days:
            day=db.load_day(date);raw=day.observations();valid=raw[raw.pv_power_w.notna()]
            t=valid.read_ts.to_numpy();median=float(np.median(np.diff(t))) if len(t)>1 else None
            report=day.energy_report(0,86400)
            fine=PVDay(date,day._array,max_gap_s=60).energy_report(0,86400)
            if len(day._times)>=2:
                intersect=(day._times[:-1]<day.start_timestamp+86400)&(day._times[1:]>day.start_timestamp)
                maxgap=float(day._dt[intersect].max()) if intersect.any() else None
                bounded=day._times[0]<=day.start_timestamp and day._times[-1]>=day.start_timestamp+86400
            else:maxgap=None;bounded=False
            recommend=int(fine.complete and median is not None and median<=15)
            con.execute('''UPDATE days SET raw_rows=?,valid_rows=?,invalid_rows=?,first_observation_s=?,last_observation_s=?,
                median_interval_s=?,max_support_gap_s=?,coverage_60_s=?,coverage_300_s=?,pv_peak_w=?,known_energy_300_j=?,
                full_day_energy_j=?,recommended=? WHERE date=?''',
                (len(raw),len(valid),len(raw)-len(valid),float(raw.read_ts.min()-day.start_timestamp),
                 float(raw.read_ts.max()-day.start_timestamp),median,maxgap if bounded else None,
                 fine.covered_seconds,report.covered_seconds,float(valid.pv_power_w.max()) if len(valid) else None,
                 report.known_energy_j,report.known_energy_j if report.complete else None,recommend,date))
    con.commit()
    head=json.loads((args.audit/'endpoint_checks.json').read_text())['head']['data']['sha']
    manifest=dict(version='1.0.0',created_date='2026-11-23',timezone='Asia/Shanghai',inverter=1,
        upstream_dataset_repository='https://github.com/OpenCEM-platform/opencem-dataset',upstream_commit=head,
        counts=dict(counts),quality_counts=quality,dates=len(days),
        date_range={'first':days[0],'last':days[-1]},
        rules=['Parse 78-column export by actual field count; skip obsolete overlapping 82-column export.',
            'Keep inverter 1 only; remove exact duplicates using full-record BLAKE2b-128 and timestamp; reject conflicting timestamps.',
            'Preserve every observed timestamp and original power text; map missing/nonfinite/negative power to NULL for interpolation.',
            'Keep zero power; do not smooth, winsorize, normalize, scale, or synthesize additional measured observations.',
            'Default interpolation calls official function; missing bracket or gap >300 seconds stays missing.',
            'Recommended days have median valid interval <=15 seconds and full 24h coverage with bracketing gaps <=60 seconds.'],
        counts_are_observations_not_interpolated_grid_points=True,sources=sources,per_source=per_source,
        official_code=json.loads((ROOT/'upstream_manifest.json').read_text()))
    for key,value in [('manifest',json.dumps(manifest)),('recommended_rule','median interval <=15s; full 24h supported with gaps <=60s')]:
        con.execute('INSERT INTO metadata VALUES (?,?)',(key,value))
    con.commit();assert con.execute('PRAGMA integrity_check').fetchone()[0]=='ok';con.close()
    tmp.replace(target)
    manifest['database_sha256']=hashlib.sha256(target.read_bytes()).hexdigest();manifest['database_bytes']=target.stat().st_size
    with PVDatabase(target) as db:
        catalog=db.days();manifest['recommended_days']=int(catalog.recommended.sum())
        catalog.to_csv(ROOT/'day_catalog.csv',index=False)
    (target.parent/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['counts','quality_counts','dates','recommended_days','database_bytes']},indent=2))

if __name__=='__main__':main()
