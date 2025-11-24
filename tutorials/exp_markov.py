import math
import random
import pandas as pd

from edge_sim_py.simulator import Simulator
from edge_sim_py.components import Topology, User


# ============================================================
# Config (simulation + economic parameters from paper)
# ============================================================
NUM_DEVICES = 20
REGION_CENTER = (0.0, 0.0)
REGION_RADIUS = 15.0          # application region
TARGET_VS_SIZE = 6            # desired VS size
SIM_STEPS = 50

GAMMA = 0.6                   # SoP smoothing
EPS_INT = 100.0               # initial energy ε_int
EPS_ACTIVE_STEP = 0.5         # energy cost per active step
EPS_THRESH = 10.0             # ε_thres (for lifetime formula if needed)

# Economic constants (same style/numbers as paper summary)
Co_Se_aas = 10.0               # Co_{Se-aas} cost per unit mSe-aaS
Co_d = 20.0                   # Co_d deployment cost per sensor
Co_p = 30.0                   # Co_p price per sensor
Co_ser = 100.0                # Co_ser monthly service cost
Co_r = 5.0                   # Co_r rent cost per unit
Co_dev = 25.0                 # Co_dev device registration cost
Co_MSC = 50.0                # Co_MSC maintenance cost for SCSP


# ============================================================
# Helpers
# ============================================================
def dist(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def random_walk(agent):
    agent.x += random.choice([-1, 0, 1])
    agent.y += random.choice([-1, 0, 1])


# ============================================================
# DeviceNode: mobile device that can be in a Virtual Sensor
# ============================================================
class DeviceNode(User):
    def __init__(self, name, x=0.0, y=0.0, v_max=3.0):
        super().__init__()
        self.name = name
        self.x = x
        self.y = y
        self.v_max = v_max

        self.sop = 0.0
        self.eps_int = EPS_INT
        self.eps_used = 0.0
        self.active_in_vs = False
        self.mobility_model = random_walk

    def position(self):
        return (self.x, self.y)

    def eps_residual(self):
        return max(0.0, self.eps_int - self.eps_used)

    def eps_res_norm(self):
        return self.eps_residual() / self.eps_int if self.eps_int > 0 else 0.0

    def in_region(self, center, radius):
        return dist(self.position(), center) <= radius

    def step(self):
        if self.mobility_model is not None:
            self.mobility_model(self)

        present = 1.0 if self.in_region(REGION_CENTER, REGION_RADIUS) else 0.0
        self.sop = GAMMA * present + (1.0 - GAMMA) * self.sop

        if self.active_in_vs:
            self.eps_used += EPS_ACTIVE_STEP

    def collect(self):
        return {
            "x": self.x,
            "y": self.y,
            "active_in_vs": int(self.active_in_vs),
            "SoP": round(self.sop, 3),
            "eps_res_norm": round(self.eps_res_norm(), 3),
        }


# ============================================================
# VSManager: manages a Virtual Sensor using Markov-style reallocation
# ============================================================
class VSManager(User):
    def __init__(self, name, center, radius, target_size, devices):
        super().__init__()
        self.name = name
        self.center = center
        self.radius = radius
        self.target_size = target_size
        self.devices = devices  # explicit list of DeviceNode agents

        self.initialized = False
        self.reallocations_this_step = 0
        self.active_count = 0
        self.inside_count = 0

    def _devices(self):
        return self.devices

    def _weighted_choice(self, candidates):
        weights = []
        for d in candidates:
            # non-equiprobable Markov reallocation:
            # weight by SoP and residual energy (paper's intuition)
            base = 0.5 * d.sop + 0.5 * d.eps_res_norm()
            w = max(1e-6, base)
            weights.append(w)
        return random.choices(candidates, weights=weights, k=1)[0]

    def _initialize_vs(self, devices):
        inside = [d for d in devices if d.in_region(self.center, self.radius)]
        if not inside:
            return
        k = min(self.target_size, len(inside))
        chosen = sorted(
            inside,
            key=lambda d: (d.sop + d.eps_res_norm()),
            reverse=True
        )[:k]
        for d in chosen:
            d.active_in_vs = True
        self.initialized = True

    def step(self):
        devices = self._devices()
        if not devices:
            return

        if not self.initialized:
            self._initialize_vs(devices)

        self.reallocations_this_step = 0

        inside = [d for d in devices if d.in_region(self.center, self.radius)]
        self.inside_count = len(inside)

        active = [d for d in devices if d.active_in_vs]
        leaving = [d for d in active if not d.in_region(self.center, self.radius)]
        candidates = [d for d in devices if (not d.active_in_vs) and d.in_region(self.center, self.radius)]

        # Markov-style reallocation of leaving devices
        for ld in leaving:
            if not candidates:
                ld.active_in_vs = False
                continue
            new_dev = self._weighted_choice(candidates)
            ld.active_in_vs = False
            new_dev.active_in_vs = True
            candidates.remove(new_dev)
            self.reallocations_this_step += 1

        # Fill VS if we have less than target_size
        active = [d for d in devices if d.active_in_vs]
        deficit = self.target_size - len(active)
        if deficit > 0 and candidates:
            add_k = min(deficit, len(candidates))
            for _ in range(add_k):
                new_dev = self._weighted_choice(candidates)
                new_dev.active_in_vs = True
                candidates.remove(new_dev)
                self.reallocations_this_step += 1

        self.active_count = len([d for d in devices if d.active_in_vs])

    def collect(self):
        devices = self._devices()
        active = [d for d in devices if d.active_in_vs]

        avg_sop = sum(d.sop for d in active) / len(active) if active else 0.0
        avg_eps = sum(d.eps_res_norm() for d in active) / len(active) if active else 0.0

        return {
            "active_vs_size": self.active_count,
            "inside_region": self.inside_count,
            "reallocations": self.reallocations_this_step,
            "avg_sop_active": round(avg_sop, 3),
            "avg_eps_res_active": round(avg_eps, 3),
        }


# ============================================================
# RMA hook (unused but required)
# ============================================================
def simple_rma(parameters=None, **kwargs):
    pass


# ============================================================
# Run simulation + print MSC-style results
# ============================================================
def run_vs_markov_simulation():
    sim = Simulator(
        stopping_criterion=lambda model: model.schedule.steps >= SIM_STEPS,
        resource_management_algorithm=simple_rma,
        tick_duration=1,
        tick_unit="seconds",
        dump_interval=1,
        logs_directory=None,
    )

    topology = sim.initialize_agent(Topology())
    sim.topology = topology

    devices = []
    for i in range(NUM_DEVICES):
        x = random.uniform(-20, 20)
        y = random.uniform(-20, 20)
        dev = sim.initialize_agent(DeviceNode(f"Device_{i+1}", x=x, y=y))
        devices.append(dev)

    vs_manager = sim.initialize_agent(
        VSManager("VSManager_1", REGION_CENTER, REGION_RADIUS, TARGET_VS_SIZE, devices)
    )

    sim.running = True
    sim.run_model()

    device_df = pd.DataFrame()
    vs_df = pd.DataFrame()

    for k in sim.agent_metrics:
        if "DeviceNode" in k:
            device_df = pd.DataFrame(sim.agent_metrics[k])
        if "VSManager" in k:
            vs_df = pd.DataFrame(sim.agent_metrics[k])

    print("\n=== VS Manager metrics (first 10 rows) ===")
    print(vs_df.head(10))

    if not vs_df.empty:
        avg_active = vs_df["active_vs_size"].mean()
        avg_realloc = vs_df["reallocations"].mean()
        print(f"\nAverage VS size over time: {avg_active:.2f}")
        print(f"Average reallocations per step: {avg_realloc:.2f}")

    print("\n=== Final snapshot of devices ===")
    if not device_df.empty:
        # last NUM_DEVICES rows = last time step for each device
        print(device_df.tail(NUM_DEVICES))

    # ========================================================
    # Extra: metrics that mirror the paper's results
    # ========================================================
    if not device_df.empty:
        # τ_a: time of activity per device (how many steps it was in VS)
        tau_a = device_df.groupby("Object")["active_in_vs"].sum()
        n1 = int(tau_a.sum())         # total mSe-aaS units equivalent
        m = tau_a.shape[0]
        avg_tau = tau_a.mean()

        # Residual energy distribution at the final time (Fig. 4-type info)
        last_step = device_df["Time Step"].max()
        last_df = device_df[device_df["Time Step"] == last_step]
        avg_eps_res_norm = last_df["eps_res_norm"].mean()
        min_eps_res_norm = last_df["eps_res_norm"].min()
        max_eps_res_norm = last_df["eps_res_norm"].max()

        print("\n=== MSC-style performance metrics (from paper's definitions) ===")
        print(f"Total service units n1 (sum of active steps τ_a): {n1}")
        print(f"Average time of activity τ_a per device: {avg_tau:.2f} steps")
        print(f"Average normalized residual energy at final step: {avg_eps_res_norm:.3f}")
        print(f"Min/Max normalized residual energy at final step: {min_eps_res_norm:.3f} / {max_eps_res_norm:.3f}")

        # === Economic / business metrics (Fig. 5 & 6 style) ===
        # Treat all devices as sensors of a single “sensor owner” and one application region.
        n2 = m  # number of sensor nodes

        # End-user cost: Co_end-user^Of = n1 * Co_Se-aaS
        Co_end_user_Of = n1 * Co_Se_aas

        # Sensor owner outflow & inflow:
        # Co_O^Of = n2 (Co_d + Co_p)
        Co_O_Of = n2 * (Co_d + Co_p)
        # Co_O^If = (Σ τ_a(j)/30) * Co_ser , Σ τ_a(j) = n1 if each step=1 day
        Co_O_If = (n1 / 30.0) * Co_ser

        # Device owner inflow: Co_D^If = Co_r * Σ τ_a(k) + Co_dev  (equation (26))
        Co_D_If = Co_r * n1 + Co_dev

        # SCSP inflow/outflow & profit (equations (29)–(30))
        m1 = 1  # single application region
        Co_SCSP_If = m1 * Co_end_user_Of
        Co_SCSP_Of = Co_O_If + Co_D_If + Co_MSC
        Profit_SCSP = Co_SCSP_If - Co_SCSP_Of

        print("\n=== MSC-style economic metrics (single-region approximation) ===")
        print(f"End-user outflow Co_end_user^Of: {Co_end_user_Of:.2f}")
        print(f"Sensor owner outflow Co_O^Of: {Co_O_Of:.2f}")
        print(f"Sensor owner inflow Co_O^If: {Co_O_If:.2f}")
        print(f"Device owner inflow Co_D^If: {Co_D_If:.2f}")
        print(f"SCSP inflow Co_SCSP^If: {Co_SCSP_If:.2f}")
        print(f"SCSP outflow Co_SCSP^Of: {Co_SCSP_Of:.2f}")
        print(f"SCSP profit: {Profit_SCSP:.2f}")


if __name__ == "__main__":
    run_vs_markov_simulation()
