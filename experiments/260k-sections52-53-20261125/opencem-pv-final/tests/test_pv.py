import hashlib,json,sqlite3,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from opencem.clock import Clock
from opencem import dataset as official
from opencem_pv import PVDatabase,PVDay,DataGapError
from opencem_pv.api import day_start

ROOT=Path(__file__).resolve().parents[1]

class InterpolationAndEnergyTests(unittest.TestCase):
    def make(self,points,**kwargs):
        start=day_start('2025-12-26')[1]
        return PVDay('2025-12-26',[(start+t,p) for t,p in points],**kwargs)

    def test_official_function_and_analytic_fractional_integral(self):
        day=self.make([(0,10),(10,20),(20,0)])
        with patch.object(official,'interpolate_value',wraps=official.interpolate_value) as called:
            np.testing.assert_allclose(day.power_at([0,2.5,10,17.5,20]),[10,12.5,20,5,0])
            self.assertAlmostEqual(day.energy(0,20),250)
            self.assertAlmostEqual(day.energy(2.5,15),215.625)
            self.assertGreaterEqual(called.call_count,7)

    def test_long_gap_and_no_endpoint_extrapolation(self):
        day=self.make([(10,100),(20,100),(1000,100)],max_gap_s=60)
        with self.assertRaises(DataGapError):day.power_at(0)
        with self.assertRaises(DataGapError):day.power_at(500)
        with self.assertRaises(DataGapError):day.energy(10,30)
        self.assertTrue(np.isnan(day.power_at(500,missing='nan')))
        report=day.energy_report(0,1030)
        self.assertAlmostEqual(report.known_energy_j,1000)
        self.assertAlmostEqual(report.covered_seconds,10)
        self.assertAlmostEqual(report.missing_seconds,1020)
        self.assertFalse(report.complete)
        explicit=self.make([(10,100),(20,100),(1000,100)],max_gap_s=None)
        self.assertEqual(explicit.power_at(500),100)
        self.assertEqual(explicit.energy(10,990),99000)

    def test_scaling_additivity_boundaries_and_zero_power(self):
        day=self.make([(0,0),(100,100),(86400,0)],max_gap_s=None,scale=.2)
        self.assertEqual(day.power_at(0),0)
        self.assertAlmostEqual(day.energy(0,86400),day.energy(0,36000)+day.energy(36000,50400),places=6)
        with self.assertRaises(ValueError):day.energy(86399,2)
        with self.assertRaises(ValueError):day.energy(0,-1)
        with self.assertRaises(ValueError):day.sample(0)
        with self.assertRaises(ValueError):day.power_at(float('nan'))
        single=self.make([(100,0)])
        self.assertEqual(single.power_at(100),0)
        with self.assertRaises(DataGapError):single.energy(100,1)

class RealDatabaseTests(unittest.TestCase):
    def setUp(self):self.db=PVDatabase()
    def tearDown(self):self.db.close()

    def test_242_days_cleaning_and_readonly(self):
        days=self.db.days();self.assertEqual(len(days),242)
        self.assertEqual(int(days.raw_rows.sum()),1135611)
        self.assertEqual(int(days.valid_rows.sum()),1134524)
        self.assertEqual(int(days.invalid_rows.sum()),1087)
        self.assertEqual(len(self.db.days(recommended_only=True)),142)
        self.assertEqual(self.db.connection.execute('PRAGMA integrity_check').fetchone()[0],'ok')
        with self.assertRaises(sqlite3.OperationalError):self.db.connection.execute("DELETE FROM days")
        with self.assertRaises(KeyError):self.db.load_day('2025-08-15')

    def test_real_day_matches_official_class(self):
        day=self.db.load_day('2025-12-26')
        clock=Clock.from_seconds(day.start_timestamp+10*3600)
        source=official.PowerSourceDataset(clock,1,self.db.connection)
        for i in range(10):
            value=source.step(17_000_000_000)
            self.assertAlmostEqual(value.power_w,day.power_at(36000+17*i),places=8)

    def test_real_day_grid_quality_and_exact_integral(self):
        day=self.db.load_day('2025-12-26');sample=day.sample(1)
        self.assertEqual(len(sample),86400);self.assertFalse(sample.pv_power_w.isna().any())
        self.assertEqual(str(sample.timestamp.dt.tz),'Asia/Shanghai')
        self.assertTrue((sample.quality=='interpolated').any())
        self.assertTrue((sample.quality=='observed').any())
        rng=np.random.default_rng(20261123)
        for _ in range(25):
            a=float(rng.uniform(0,85000));duration=float(rng.uniform(10,650));b=a+duration
            knots=day._times[(day._times>day.start_timestamp+a)&(day._times<day.start_timestamp+b)]
            ts=np.r_[day.start_timestamp+a,knots,day.start_timestamp+b]
            values=np.interp(ts,day._times,day._values)
            expected=np.sum(np.diff(ts)*.5*(values[:-1]+values[1:]))
            self.assertAlmostEqual(day.energy(a,duration),expected,delta=1e-5)
        partial=self.db.load_day('2025-07-12').sample(60)
        self.assertTrue(partial.pv_power_w.isna().any())
        self.assertTrue((partial.quality=='outside_coverage').any())

    def test_all_dates_are_loadable_and_recommended_dates_have_coverage(self):
        for row in self.db.days().itertuples():
            day=self.db.load_day(row.date)
            self.assertEqual(len(day.observations()),row.raw_rows)
            self.assertTrue(day.observations().read_ts.is_monotonic_increasing)
            if row.recommended:
                self.assertTrue(self.db.load_day(row.date,max_gap_s=60).energy_report(0,86400).complete)

    def test_upstream_files_are_unmodified(self):
        for row in json.loads((ROOT/'upstream_manifest.json').read_text()):
            content=(ROOT/row['path']).read_bytes()
            actual=hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()
            self.assertEqual(actual,row['git_blob_sha1'])

if __name__=='__main__':unittest.main()
