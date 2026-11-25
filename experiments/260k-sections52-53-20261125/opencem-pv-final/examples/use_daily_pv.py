"""Run from the distribution root with PYTHONPATH=src python examples/use_daily_pv.py."""
from opencem_pv import PVDatabase

with PVDatabase() as db:
    print(db.days(recommended_only=True)[['date','raw_rows','median_interval_s']].head())
    day=db.load_day('2025-12-26')
    native=day.observations()        # Real, unscaled measurements and quality flags.
    grid=day.sample(step_s=1)        # Official interpolation, not additional measurements.
    print(grid.head())
    print('PV at 10:00:',day.power_at(10*3600),'W')
    print('PV energy of a 185.8s epoch at 10:00:',day.energy(10*3600,185.8),'J')
    scaled=db.load_day('2025-12-26',scale=.1)
    print('Same epoch, PV scaled to 10%:',scaled.energy(10*3600,185.8),'J')
