"""Read-only PV data access and exact integration of official linear interpolation.

All dates use Asia/Shanghai; offsets and durations use seconds; power uses W;
energy uses J. No day wrapping, endpoint extrapolation, or implicit gap filling.
"""
from dataclasses import dataclass
from datetime import date as Date, datetime, time
from pathlib import Path
import json
import math
import sqlite3
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from opencem import dataset as official

TIMEZONE = ZoneInfo("Asia/Shanghai")
DEFAULT_DATABASE = Path(__file__).resolve().parent / "data" / "opencem_pv.sqlite"
DEFAULT_MAX_GAP_S = 300.0


class DataGapError(ValueError):
    """The requested power or energy is not fully supported by observations."""


def day_start(date):
    day = Date.fromisoformat(str(date))
    return str(day), datetime.combine(day, time(), tzinfo=TIMEZONE).timestamp()


@dataclass(frozen=True)
class EnergyReport:
    known_energy_j: float
    requested_seconds: float
    covered_seconds: float
    missing_seconds: float
    complete: bool

    @property
    def coverage_fraction(self):
        return self.covered_seconds / self.requested_seconds if self.requested_seconds else 1.0


class PVDay:
    """One calendar day, including adjacent observations for midnight interpolation.

    ``max_gap_s=None`` explicitly enables the official unlimited-gap interpolation
    inside the observed range. The default rejects intervals longer than 300 s.
    All 242 observed dates remain available, including incomplete/coarse dates.
    """

    def __init__(self, date, array, observations=None, *, max_gap_s=DEFAULT_MAX_GAP_S, scale=1.0, metadata=None):
        self.date, self.start_timestamp = day_start(date)
        self.scale = float(scale)
        if not np.isfinite(self.scale) or self.scale < 0:
            raise ValueError("scale must be finite and nonnegative")
        if max_gap_s is not None and (not np.isfinite(max_gap_s) or max_gap_s <= 0):
            raise ValueError("max_gap_s must be positive, or None for explicit unlimited official interpolation")
        self.max_gap_s = max_gap_s
        self.metadata = dict(metadata or {})
        self._raw = observations.copy() if observations is not None else pd.DataFrame()
        self._array = np.asarray(array, dtype=float).reshape(-1, 2).copy()
        if len(self._array) and (not np.isfinite(self._array).all() or
                                np.any(np.diff(self._array[:, 0]) <= 0) or
                                np.any(self._array[:, 1] < 0)):
            raise ValueError("Interpolation input must have unique sorted times and finite nonnegative power")
        self._array.setflags(write=False)
        self._times = self._array[:, 0]
        self._values = self._array[:, 1]
        self._dt = np.diff(self._times)
        self._good = self._dt <= (float('inf') if max_gap_s is None else max_gap_s)
        areas = 0.5 * (self._values[:-1] + self._values[1:]) * self._dt
        self._energy_prefix = np.r_[0.0, np.cumsum(np.where(self._good, areas, 0.0))]
        self._coverage_prefix = np.r_[0.0, np.cumsum(np.where(self._good, self._dt, 0.0))]

    @staticmethod
    def _offsets(seconds):
        values = np.asarray(seconds, dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 86400):
            raise ValueError("seconds must be finite and within [0, 86400]; days never wrap")
        return values

    def observations(self):
        """Return unique real observations, including invalid originals and quality flags.

        These remain in native, unscaled W; scale is applied only to sampled
        trajectories and energy. No interpolated point is stored in this table.
        """
        return self._raw.copy()

    def _evaluate(self, seconds):
        offsets = self._offsets(seconds)
        ts = self.start_timestamp + offsets.ravel()
        power = np.full(len(ts), np.nan)
        gaps = np.full(len(ts), np.nan)
        quality = np.full(len(ts), "outside_coverage", dtype="<U20")
        if not len(self._times):
            return power, quality, gaps
        idx = np.searchsorted(self._times, ts)
        exact = (idx < len(self._times)) & (self._times[np.minimum(idx, len(self._times)-1)] == ts)
        bounded = (ts >= self._times[0]) & (ts <= self._times[-1])
        quality[exact] = "observed"
        gaps[exact] = 0.0
        if len(self._times) >= 2:
            interval = np.clip(idx-1, 0, len(self._times)-2)
            interp = bounded & ~exact
            gaps[interp] = self._dt[interval[interp]]
            allowed = interp & self._good[interval]
            quality[interp] = "long_gap"
            quality[allowed] = "interpolated"
        else:
            allowed = np.zeros(len(ts), dtype=bool)
        for i in np.flatnonzero(exact | allowed):
            # Deliberately call the unmodified, pinned upstream function.
            power[i] = official.interpolate_value(self._array, float(ts[i])) * self.scale
        return power, quality, gaps

    def power_at(self, seconds, *, missing="raise"):
        """Get scalar/array power at second offsets from local midnight."""
        if missing not in ("raise", "nan"):
            raise ValueError("missing must be 'raise' or 'nan'")
        shape = np.asarray(seconds).shape
        power, quality, _ = self._evaluate(seconds)
        if missing == "raise" and np.isnan(power).any():
            raise DataGapError(f"{self.date}: query includes {sorted(set(quality[np.isnan(power)]))}; "
                               "inspect sample()/metadata or choose a covered interval")
        return float(power[0]) if shape == () else power.reshape(shape)

    def sample(self, step_s=1.0):
        """Return [00:00, 24:00) grid points; missing regions stay NaN and labeled.

        One-second output is an interpolation representation, not 1 Hz sensing.
        """
        if not np.isfinite(step_s) or step_s <= 0 or step_s > 86400:
            raise ValueError("step_s must be in (0, 86400]")
        if math.ceil(86400 / step_s) > 2_000_000:
            raise ValueError("Grid too large; query a short interval with power_at instead")
        seconds = np.arange(0.0, 86400.0, float(step_s))
        power, quality, gap = self._evaluate(seconds)
        result = pd.DataFrame(dict(second_of_day=seconds,
            timestamp=pd.to_datetime(self.start_timestamp + seconds, unit="s", utc=True).tz_convert(TIMEZONE),
            pv_power_w=power, quality=quality, support_interval_s=gap))
        result.attrs.update(date=self.date, timezone=str(TIMEZONE), scale=self.scale,
            interpolation="opencem.dataset.interpolate_value", max_gap_s=self.max_gap_s)
        return result

    def _cumulative(self, ts):
        if len(self._times) < 2 or ts <= self._times[0]:
            return 0.0, 0.0
        if ts >= self._times[-1]:
            return float(self._energy_prefix[-1]), float(self._coverage_prefix[-1])
        i = int(np.searchsorted(self._times, ts, side="right")-1)
        energy, covered = self._energy_prefix[i], self._coverage_prefix[i]
        if self._good[i]:
            # Exact integral under the official linear curve; no one-second
            # quadrature error, including fractional epoch start/end times.
            value = official.interpolate_value(self._array, float(ts))
            elapsed = ts-self._times[i]
            energy += 0.5*(self._values[i]+value)*elapsed
            covered += elapsed
        return float(energy), float(covered)

    def energy_report(self, start_s, duration_s):
        """Report known energy and coverage; never describe partial energy as full."""
        start_s, duration_s = float(start_s), float(duration_s)
        if not np.isfinite(duration_s) or duration_s < 0:
            raise ValueError("duration_s must be finite and nonnegative")
        self._offsets([start_s, start_s+duration_s])
        e0, c0 = self._cumulative(self.start_timestamp+start_s)
        e1, c1 = self._cumulative(self.start_timestamp+(start_s+duration_s))
        covered = min(duration_s, max(0.0, c1-c0))
        missing = max(0.0, duration_s-covered)
        return EnergyReport((e1-e0)*self.scale, duration_s, covered, missing, missing < 1e-6)

    def energy(self, start_s, duration_s):
        """PV energy in J for one epoch; raises if its interval crosses a data gap."""
        report = self.energy_report(start_s, duration_s)
        if not report.complete:
            raise DataGapError(f"{self.date}: {report.missing_seconds:.6f} of "
                               f"{duration_s} seconds lack support (max_gap_s={self.max_gap_s})")
        return report.known_energy_j

    def energy_many(self, starts_s, durations_s):
        starts, durations = np.broadcast_arrays(starts_s, durations_s)
        return np.array([self.energy(s, t) for s, t in zip(starts.ravel(), durations.ravel())]).reshape(starts.shape)


class PVDatabase:
    """Local read-only SQLite database; no network or GPU access at runtime."""

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else DEFAULT_DATABASE
        if not self.path.is_file():
            raise FileNotFoundError(f"PV database missing: {self.path}")
        self.connection = sqlite3.connect(self.path.resolve().as_uri()+"?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row
        meta = self.connection.execute("SELECT value FROM metadata WHERE key='format_version'").fetchone()
        if meta is None or meta[0] != '1':
            self.close()
            raise ValueError("Unsupported PV database format")

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def days(self, *, recommended_only=False):
        """List all 242 dates, or the predeclared quality-only experimental subset."""
        where = " WHERE recommended=1" if recommended_only else ""
        return pd.read_sql_query("SELECT * FROM days"+where+" ORDER BY date", self.connection)

    def load_day(self, date, *, max_gap_s=DEFAULT_MAX_GAP_S, scale=1.0):
        date, start = day_start(date)
        row = self.connection.execute("SELECT * FROM days WHERE date=?", (date,)).fetchone()
        if row is None:
            raise KeyError(f"{date} has no observations; use days() for available dates")
        raw = pd.read_sql_query("SELECT * FROM pv_observations WHERE read_ts>=? AND read_ts<? ORDER BY read_ts",
                                self.connection, params=(start,start+86400))
        raw['timestamp'] = pd.to_datetime(raw.read_ts,unit='s',utc=True).dt.tz_convert(TIMEZONE)
        rows = self.connection.execute("""
            SELECT read_ts,pv_power_w FROM pv_observations WHERE pv_power_w IS NOT NULL
            AND read_ts>=COALESCE((SELECT MAX(read_ts) FROM pv_observations
                WHERE pv_power_w IS NOT NULL AND read_ts<=?), ?)
            AND read_ts<=COALESCE((SELECT MIN(read_ts) FROM pv_observations
                WHERE pv_power_w IS NOT NULL AND read_ts>=?), ?) ORDER BY read_ts
            """, (start,start,start+86400,start+86400)).fetchall()
        return PVDay(date, np.asarray([tuple(r) for r in rows],dtype=float).reshape(-1,2), raw,
                     max_gap_s=max_gap_s,scale=scale,metadata=dict(row))
