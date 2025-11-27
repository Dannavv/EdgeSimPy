import math
import random
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from edge_sim_py.simulator import Simulator
from edge_sim_py.components import (
    EdgeServer,
    User,
    BaseStation,
    NetworkSwitch,
    NetworkLink,
    Topology,
)
from edge_sim_py.activation_schedulers import DefaultScheduler


def dummy_flow_scheduler(topology, flows):
    return


def dist(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def clamp(x, low, high):
    return max(low, min(high, x))


# -----------------------------------------------------
# ------------------ CUSTOM COMPONENTS -----------------
# -----------------------------------------------------

class MonitoredBaseStation(BaseStation):
    def collect(self) -> dict:
        return {
            "Instance ID": self.id,
            "Coordinates": self.coordinates,
            "Wireless Delay": getattr(self, "wireless_delay", None),
            "Num Users": len(self.users),
        }


class CloudServer(EdgeServer):
    def __init__(self, obj_id=None, mips=20000, ram=128000, storage=200000):
        super().__init__(obj_id=obj_id,
                         coordinates=None,
                         model_name="Cloud",
                         cpu=mips,
                         memory=ram,
                         disk=storage)
        self.name = "Cloud"
        self.mips = mips
        self.ram = ram
        self.storage = storage

        self.request_load = 0
        self.cpu_util = 0.0
        self.energy = 0.0

        self.total_requests = 0
        self.total_delay = 0.0
        self.total_net_delay = 0.0
        self.total_proc_delay = 0.0

    def receive_data(self, data_size, user):
        wireless = user.base_station.wireless_delay if user.base_station else 0.0

        topo = self.model.topology
        if user.base_station and self.base_station:
            path_switches = nx.shortest_path(
                G=topo,
                source=user.base_station.network_switch,
                target=self.base_station.network_switch,
                weight="delay",
            )
            path_delay = topo.calculate_path_delay(path_switches)
        else:
            path_delay = 0.0

        net_delay = wireless + path_delay

        self.request_load += data_size
        self.cpu_util = min(1.0, self.request_load / (self.mips + 1e3))

        proc_delay = data_size * 0.02 / (self.mips + 1e-9)
        total_delay = net_delay + proc_delay

        self.energy += data_size * 0.001
        self.total_requests += 1
        self.total_delay += total_delay
        self.total_net_delay += net_delay
        self.total_proc_delay += proc_delay

        return {
            "layer": "cloud",
            "server": self.name,
            "delay": total_delay,
            "net_delay": net_delay,
            "proc_delay": proc_delay,
        }

    def collect(self) -> dict:
        avg_delay = self.total_delay / self.total_requests if self.total_requests else 0
        avg_net_delay = self.total_net_delay / self.total_requests if self.total_requests else 0
        avg_proc_delay = self.total_proc_delay / self.total_requests if self.total_requests else 0

        return {
            "Instance ID": self.id,
            "Name": self.name,
            "Layer": "cloud",
            "CPU": self.cpu,
            "RAM": self.memory,
            "Disk": self.disk,
            "Request Load": self.request_load,
            "CPU Util": self.cpu_util,
            "Energy": self.energy,
            "Total Requests": self.total_requests,
            "Avg Delay": avg_delay,
            "Avg Net Delay": avg_net_delay,
            "Avg Proc Delay": avg_proc_delay,
        }


class FogServer(EdgeServer):
    def __init__(self, name, obj_id=None, mips=8000, ram=64000, storage=100000):
        super().__init__(obj_id=obj_id,
                         coordinates=None,
                         model_name=name,
                         cpu=mips,
                         memory=ram,
                         disk=storage)
        self.name = name
        self.mips = mips
        self.ram = ram
        self.storage = storage

        self.request_load = 0
        self.cpu_util = 0.0
        self.energy = 0.0

        self.total_requests = 0
        self.total_delay = 0.0
        self.total_net_delay = 0.0
        self.total_proc_delay = 0.0

    def receive_data(self, data_size, user):
        wireless = user.base_station.wireless_delay if user.base_station else 0.0

        topo = self.model.topology
        if user.base_station and self.base_station:
            path_switches = nx.shortest_path(
                G=topo,
                source=user.base_station.network_switch,
                target=self.base_station.network_switch,
                weight="delay",
            )
            path_delay = topo.calculate_path_delay(path_switches)
        else:
            path_delay = 0.0

        net_delay = wireless + path_delay

        self.request_load += data_size
        self.cpu_util = min(1.0, self.request_load / (self.mips + 1e3))

        proc_delay = data_size * 0.05 / (self.mips + 1e-9)
        total_delay = net_delay + proc_delay

        self.energy += data_size * 0.0005
        self.total_requests += 1
        self.total_delay += total_delay
        self.total_net_delay += net_delay
        self.total_proc_delay += proc_delay

        return {
            "layer": "fog",
            "server": self.name,
            "delay": total_delay,
            "net_delay": net_delay,
            "proc_delay": proc_delay,
        }

    def collect(self) -> dict:
        avg_delay = self.total_delay / self.total_requests if self.total_requests else 0
        avg_net_delay = self.total_net_delay / self.total_requests if self.total_requests else 0
        avg_proc_delay = self.total_proc_delay / self.total_requests if self.total_requests else 0

        return {
            "Instance ID": self.id,
            "Name": self.name,
            "Layer": "fog",
            "CPU": self.cpu,
            "RAM": self.memory,
            "Disk": self.disk,
            "Request Load": self.request_load,
            "CPU Util": self.cpu_util,
            "Energy": self.energy,
            "Total Requests": self.total_requests,
            "Avg Delay": avg_delay,
            "Avg Net Delay": avg_net_delay,
            "Avg Proc Delay": avg_proc_delay,
        }


# -----------------------------------------------------
# ----------------------- USERS -----------------------
# -----------------------------------------------------

class MovingUser(User):

    def __init__(self, obj_id=None,
                 area_width=800,
                 area_height=200,
                 data_size_min=100,
                 data_size_max=1000,
                 move_speed=1):
        super().__init__(obj_id=obj_id)
        self.x = 0
        self.y = 0
        self.area_width = area_width
        self.area_height = area_height
        self.data_size_min = data_size_min
        self.data_size_max = data_size_max
        self.move_speed = move_speed

        self.total_requests = 0
        self.sla_violations = 0
        self.sla_violation_rate = 0.0

        self.last_delay = 0.0
        self.last_server_name = None
        self.last_server_layer = None

    def _init_position(self):
        self.x = random.randint(0, self.area_width)
        self.y = random.randint(0, self.area_height)
        self.coordinates = [self.x, self.y]

        all_bs = BaseStation.all()
        nearest = min(all_bs, key=lambda bs: dist(self.coordinates, bs.coordinates))
        self.base_station = nearest
        nearest.users.append(self)

    def _move(self):
        self.x += random.randint(-self.move_speed, self.move_speed)
        self.y += random.randint(-self.move_speed, self.move_speed)
        self.x = clamp(self.x, 0, self.area_width)
        self.y = clamp(self.y, 0, self.area_height)
        self.coordinates = [self.x, self.y]

        bss = BaseStation.all()
        nearest = min(bss, key=lambda bs: dist(self.coordinates, bs.coordinates))
        if self.base_station is not nearest:
            if self.base_station and self in self.base_station.users:
                self.base_station.users.remove(self)
            nearest.users.append(self)
            self.base_station = nearest

    # -----------------------------------------------------
    # --------- MARKOV PREDICTION FOR SERVER --------------
    # -----------------------------------------------------
    def _predict_server_markov(self):
        if self.last_server_name is None:
            return None

        M = self.model.markov_matrix
        if self.last_server_name not in M:
            return None

        transitions = M[self.last_server_name]
        servers = list(transitions.keys())
        probs = list(transitions.values())

        return random.choices(servers, probs)[0]

    def _select_server(self):
        fogs = self.model.fog_servers
        cloud = self.model.cloud_server
        fog_names = [f.name for f in fogs]

        # Markov prediction
        predicted = self._predict_server_markov()

        if predicted in fog_names:
            fog = next(f for f in fogs if f.name == predicted)
            d = dist(self.coordinates, fog.base_station.coordinates)
            if d <= self.model.max_fog_distance:
                return fog, d

        # fallback nearest fog
        nearest_fog = min(fogs, key=lambda f: dist(self.coordinates, f.base_station.coordinates))
        d = dist(self.coordinates, nearest_fog.base_station.coordinates)

        if d <= self.model.max_fog_distance:
            return nearest_fog, d

        return cloud, d

    def step(self):
        if self.base_station is None:
            self._init_position()

        self._move()
        data_size = random.randint(self.data_size_min, self.data_size_max)

        server, d = self._select_server()
        stats = server.receive_data(data_size, self)

        self.last_server_name = stats["server"]
        self.last_server_layer = stats["layer"]

        sla_violated = stats["delay"] > self.model.delay_sla
        self.total_requests += 1
        if sla_violated:
            self.sla_violations += 1
        self.sla_violation_rate = self.sla_violations / self.total_requests

        self.last_delay = stats["delay"]

    def collect(self):
        return {
            "Instance ID": self.id,
            "Coordinates": self.coordinates,
            "Base Station": self.base_station.id if self.base_station else None,
            "Last Server": self.last_server_name,
            "Last Delay": self.last_delay,
            "SLA Violation Rate": self.sla_violation_rate,
        }


# -----------------------------------------------------
# ---------------- SIM, INFRASTRUCTURE ----------------
# -----------------------------------------------------

def noop_resource_management(parameters: dict):
    return


def stopping_criterion(model: Simulator) -> bool:
    return model.schedule.steps >= model.max_steps


def build_infrastructure(sim: Simulator,
                         fog_coords=None,
                         wireless_delay=5.0,
                         link_delay=5.0,
                         link_bandwidth=10000):

    if fog_coords is None:
        fog_coords = [(0, 0), (250, 0), (500, 0), (750, 0)]

    topo = sim.initialize_agent(Topology())
    sim.topology = topo

    switches = []
    fog_servers = []

    for i, (x, y) in enumerate(fog_coords, start=1):
        bs = MonitoredBaseStation()
        sim.initialize_agent(bs)
        bs.coordinates = [x, y]
        bs.wireless_delay = wireless_delay

        sw = NetworkSwitch()
        sim.initialize_agent(sw)
        bs._connect_to_network_switch(sw)
        topo.add_node(sw)

        fog = FogServer(name=f"Fog-{i}")
        sim.initialize_agent(fog)
        bs._connect_to_edge_server(fog)

        switches.append(sw)
        fog_servers.append(fog)

    cloud = CloudServer()
    sim.initialize_agent(cloud)
    BaseStation.all()[0]._connect_to_edge_server(cloud)

    for i in range(len(switches) - 1):
        sw1 = switches[i]
        sw2 = switches[i + 1]
        link = NetworkLink()
        sim.initialize_agent(link)
        link.topology = topo
        link.nodes = [sw1, sw2]
        link.delay = link_delay
        link.bandwidth = link_bandwidth

        topo.add_edge(sw1, sw2)
        topo._adj[sw1][sw2] = link
        topo._adj[sw2][sw1] = link

        sw1.links.append(link)
        sw2.links.append(link)

    sim.fog_servers = fog_servers
    sim.cloud_server = cloud

    return fog_servers


def create_users(sim, n_users=50, area_width=800, area_height=200):
    users = []
    for i in range(1, n_users + 1):
        u = MovingUser(obj_id=i,
                       area_width=area_width,
                       area_height=area_height,
                       move_speed=10)
        sim.initialize_agent(u)
        users.append(u)
    return users


# -----------------------------------------------------
# --------------------- VISUALIZATION -----------------
# -----------------------------------------------------

def plot_server_cpu(df_servers, ax):
    if df_servers.empty:
        return
    for name, grp in df_servers.groupby("Name"):
        ax.plot(grp["Time Step"], grp["CPU Util"], label=name)
    ax.set_title("Server CPU Utilization over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("CPU Util")
    ax.legend()
    ax.grid(True)


def plot_server_energy(df_servers, ax):
    if df_servers.empty:
        return
    for name, grp in df_servers.groupby("Name"):
        ax.plot(grp["Time Step"], grp["Energy"], label=name)
    ax.set_title("Server Energy over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Energy")
    ax.legend()
    ax.grid(True)


def plot_avg_delay_by_layer(df_servers, ax):
    if df_servers.empty:
        return
    df = df_servers.groupby(["Time Step", "Layer"])["Avg Delay"].mean().reset_index()
    for layer, grp in df.groupby("Layer"):
        ax.plot(grp["Time Step"], grp["Avg Delay"], label=layer)
    ax.set_title("Average Delay per Layer")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Avg Delay")
    ax.legend()
    ax.grid(True)


def plot_sla_violation_rate(df_users, ax):
    if df_users.empty:
        return
    df = df_users.groupby("Time Step")["SLA Violation Rate"].mean().reset_index()
    ax.plot(df["Time Step"], df["SLA Violation Rate"])
    ax.set_title("Average SLA Violation Rate (Users)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("SLA Violation Rate")
    ax.grid(True)


def plot_bs_load(df_bss, ax):
    if df_bss.empty:
        return
    for bs_id, grp in df_bss.groupby("Instance ID"):
        ax.plot(grp["Time Step"], grp["Num Users"], label=f"BS-{bs_id}")
    ax.set_title("Number of Users per Base Station")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Users")
    ax.legend()
    ax.grid(True)


def plot_reallocations(df_users, ax):
    if df_users.empty:
        return

    df_users["PrevBS"] = df_users.groupby("Instance ID")["Base Station"].shift(1)
    realloc = df_users[df_users["Base Station"] != df_users["PrevBS"]]

    counts = realloc.groupby("Time Step").size().reset_index(name="handoffs")

    ax.plot(counts["Time Step"], counts["handoffs"])
    ax.set_title("User Handoffs Per Time Step")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Handoffs")
    ax.grid(True)



# -----------------------------------------------------
# --------------------- MAIN EXPERIMENT ---------------
# -----------------------------------------------------

def run_experiment_markov_edgesimpy(
    n_users=50,
    max_steps=500,
    max_fog_distance=200.0,
    delay_sla=100.0,
):

    random.seed(0)

    sim = Simulator(
        stopping_criterion=stopping_criterion,
        resource_management_algorithm=noop_resource_management,
        resource_management_algorithm_parameters={},
        user_defined_functions=[],
        network_flow_scheduling_algorithm=dummy_flow_scheduler,
        tick_duration=1,
        tick_unit="seconds",
        scheduler=DefaultScheduler,
        dump_interval=float("inf"),
        logs_directory="logs_markov_edgesimpy",
    )

    sim.max_steps = max_steps
    sim.max_fog_distance = max_fog_distance
    sim.delay_sla = delay_sla

    fog_servers = build_infrastructure(sim)
    users = create_users(sim, n_users)

    # -----------------------------------------------------
    # --------------------- MARKOV MATRIX -----------------
    # -----------------------------------------------------
    sim.markov_matrix = {
        "Fog-1": {"Fog-1": 0.7, "Fog-2": 0.2, "Cloud": 0.1},
        "Fog-2": {"Fog-2": 0.7, "Fog-1": 0.1, "Fog-3": 0.2},
        "Fog-3": {"Fog-3": 0.7, "Fog-2": 0.2, "Fog-4": 0.1},
        "Fog-4": {"Fog-4": 0.6, "Fog-3": 0.3, "Cloud": 0.1},
        "Cloud": {"Cloud": 0.8, "Fog-1": 0.2},
    }

    sim.run_model()

    df_servers = pd.DataFrame(
        sim.agent_metrics.get("CloudServer", []) +
        sim.agent_metrics.get("FogServer", [])
    )
    df_users = pd.DataFrame(sim.agent_metrics.get("MovingUser", []))
    df_bs = pd.DataFrame(sim.agent_metrics.get("MonitoredBaseStation", []))

    return sim, df_servers, df_users, df_bs


if __name__ == "__main__":
    sim, df_srv, df_usr, df_bs = run_experiment_markov_edgesimpy()

    df_srv.to_csv("experiment_markov_servers_built_in.csv", index=False)
    df_usr.to_csv("experiment_markov_users_built_in.csv", index=False)
    df_bs.to_csv("experiment_markov_basestations_built_in.csv", index=False)

    print("Saved CSVs for servers, users, base stations.")

    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    fig.suptitle("Simulation Results Dashboard (Markov Model)", fontsize=16)

    plot_server_cpu(df_srv, axes[0, 0])
    plot_server_energy(df_srv, axes[0, 1])
    plot_avg_delay_by_layer(df_srv, axes[1, 0])
    plot_sla_violation_rate(df_usr, axes[1, 1])
    plot_bs_load(df_bs, axes[2, 0])
    plot_reallocations(df_usr, axes[2, 1])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("simulation_dashboard_markov.png")
    print("Saved dashboard → simulation_dashboard_markov.png")

    plt.show()
