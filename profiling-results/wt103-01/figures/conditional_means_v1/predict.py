"""Evaluate the no-PV GPU profile on the *observed* decision-state support."""
import csv
from pathlib import Path

class Profile:
    def __init__(self, directory, horizon=10, backend="math"):
        p = Path(directory)
        def get(name):
            with (p / name).open(newline="") as f:
                return [r for r in csv.DictReader(f) if r["backend"] == backend and int(r["horizon"]) == horizon]
        self.demand = {int(r["batch"]):r for r in get("moment_coefficients.csv")
                       if r["target"] == "D_J" and r["form"] == "B_J"}
        self.cells = {(int(r["batch"]),int(r["phase"])):r for r in get("bounded_epsilon.csv") if int(r["phase"])>0}
        if not self.cells or not self.demand:
            raise ValueError("No fitted profile for this horizon/backend")
    def __call__(self, J, phase, batch):
        cell=self.cells.get((batch,phase))
        if cell is None:
            raise ValueError("Unobserved (batch, phase)")
        if not float(cell["supported_J_min"])-1e-12 <= J <= float(cell["supported_J_max"])+1e-12:
            raise ValueError("J outside observed range for this (batch, phase)")
        demand=self.demand[batch]
        duration=float(cell["mean_T_s"])
        joules=float(demand["intercept"])+float(demand["coef_J_minus_1"])*(J-1)
        rho=float(cell["time_weighted_retention"])
        if not (joules>0 and duration>0 and 0<=rho<=1):
            raise ValueError("Invalid fit")
        return dict(m_T_seconds=duration,m_GPU_demand_joules=joules,
                    m_epsilon_seconds=duration*J*rho)
