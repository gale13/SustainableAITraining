"""Verify the release, redraw its three current figures, or run saved pipelines."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CURRENT=['260k-section52-mean-observed-pv','260k-section53-v-mean-observed-pv',
         '260k-section53-weight-sameV','260k-section53-weight-sameV-increasing']


def run(name,*args):
    subprocess.run([sys.executable,*args],cwd=ROOT/name,check=True)


def verify():
    manifest=json.loads((ROOT/'bundle_manifest.json').read_text())
    for n,h in manifest.items():
        p=ROOT/n
        if not p.exists():raise SystemExit(f'Missing {n}. Run python download_data.py or unpack the full release ZIP here.')
        assert hashlib.sha256(p.read_bytes()).hexdigest()==h,n
    for name in CURRENT:
        code="import experiment; p=experiment.checked_protocol() if hasattr(experiment,'checked_protocol') else experiment.checked(); print('Verified protocol:',p.get('name','Section 5.2'))"
        run(name,'-c',code)
    print('All release hashes and current protocol dependencies verified.')


def figures():
    run(CURRENT[0],'plot_figure8.py')
    run(CURRENT[1],'plot_figure.py')
    run('260k-section53-weight-sameV-ab','plot_figure.py')
    print('Rebuilt the three current figures from existing source tables; no simulations rerun.')


def analyze():
    # Re-analyze saved runs without rewriting immutable raw input archives.
    verify()
    for name in CURRENT:
        run(name,'analyze.py')
        if name==CURRENT[0]:run(name,'prepare_figure_data.py')
    figures()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['verify','figures','analyze'])
    globals()[p.parse_args().stage]()
