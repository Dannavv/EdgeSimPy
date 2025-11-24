import math
import random
import pandas as pd
import networkx as nx

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
    # We don't use NetworkFlow in this experiment,
    # so this can safely do nothing.
    return


# ---------- helpers ----------

def dist(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def clamp(x, low, high):
    return max(low, min(high, x))


# ---------- custom components ----------

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

    def receive_data(self, data_size, user):
        # Wireless delay from user's base station
        wireless = user.base_station.wireless_delay if user.base_station else 0.0

        # Network delay along the wired path (via NetworkSwitches)
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

        # CPU load / processing delay as before
        self.request_load += data_size
        self.cpu_util = min(1.0, self.request_load / (self.mips + 1e3))
        proc_delay = data_size * 0.02 / (self.mips + 1e-9)

        total_delay = net_delay + proc_delay
        self.energy += data_size * 0.001

        return {
            "layer": "cloud",
            "server": self.name,
            "delay": total_delay,
            "net_delay": net_delay,
            "proc_delay": proc_delay,
        }

    def collect(self) -> dict:
        return {
            "Instance ID": self.id,
            "Name": self.name,
            "Layer": "cloud",
            "Coordinates": self.coordinates,
            "CPU": self.cpu,
            "RAM": self.memory,
            "Disk": self.disk,
            "Request Load": self.request_load,
            "CPU Util": self.cpu_util,
            "Energy": self.energy,
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

        return {
            "layer": "fog",
            "server": self.name,
            "delay": total_delay,
            "net_delay": net_delay,
            "proc_delay": proc_delay,
        }

    def collect(self) -> dict:
        return {
            "Instance ID": self.id,
            "Name": self.name,
            "Layer": "fog",
            "Coordinates": self.coordinates,
            "CPU": self.cpu,
            "RAM": self.memory,
            "Disk": self.disk,
            "Request Load": self.request_load,
            "CPU Util": self.cpu_util,
            "Energy": self.energy,
        }


class MovingUser(User):
    """User whose step() does: mobility + request generation."""

    def __init__(self, obj_id=None,
                 area_width=800,
                 area_height=200,
                 data_size_min=100,
                 data_size_max=1000):
        super().__init__(obj_id=obj_id)
        # external 2D coordinates for our logic
        self.x = 0
        self.y = 0
        self.area_width = area_width
        self.area_height = area_height
        self.data_size_min = data_size_min
        self.data_size_max = data_size_max

    def _init_position(self):
        self.x = random.randint(0, self.area_width)
        self.y = random.randint(0, self.area_height)
        self.coordinates = [self.x, self.y]

        # connect to nearest base station
        bss = BaseStation.all()
        nearest = min(bss, key=lambda bs: dist(self.coordinates, bs.coordinates))
        self.base_station = nearest
        nearest.users.append(self)

    def _move(self):
        self.x += random.choice([-1, 0, 1])
        self.y += random.choice([-1, 0, 1])
        self.x = clamp(self.x, 0, self.area_width)
        self.y = clamp(self.y, 0, self.area_height)
        self.coordinates = [self.x, self.y]

        # re-attach to nearest base station
        bss = BaseStation.all()
        nearest = min(bss, key=lambda bs: dist(self.coordinates, bs.coordinates))
        if self.base_station is not nearest:
            if self.base_station and self in self.base_station.users:
                self.base_station.users.remove(self)
            self.base_station = nearest
            nearest.users.append(self)

    def _select_server(self):
        fog_servers = self.model.fog_servers
        cloud = self.model.cloud_server
        max_fog_distance = self.model.max_fog_distance

        # distance from user to each fog base station
        ux, uy = self.x, self.y
        nearest_fog = min(
            fog_servers,
            key=lambda f: dist((ux, uy), tuple(f.base_station.coordinates))
        )
        d = dist((ux, uy), tuple(nearest_fog.base_station.coordinates))

        if d <= max_fog_distance:
            return nearest_fog, d
        return cloud, d  # d is to nearest fog; you can also compute to cloud if you like

    def step(self):
        # first-time init
        if self.base_station is None:
            self._init_position()

        # mobility
        self._move()

        # traffic
        data_size = random.randint(self.data_size_min, self.data_size_max)
        server, d = self._select_server()
        stats = server.receive_data(data_size, self)

        t = self.model.schedule.steps  # current time step

        delay_sla = self.model.delay_sla
        sla_violated = stats["delay"] > delay_sla

        self.model.request_records.append(
            {
                "t": t,
                "user_id": self.id,
                "user_x": self.x,
                "user_y": self.y,
                "server_layer": stats["layer"],
                "server_name": stats["server"],
                "server_x": server.base_station.coordinates[0],
                "server_y": server.base_station.coordinates[1],
                "distance_to_nearest_fog": d,
                "data_size": data_size,
                "total_delay": stats["delay"],
                "net_delay": stats["net_delay"],
                "proc_delay": stats["proc_delay"],
                "server_cpu_util": server.cpu_util,
                "server_energy": server.energy,
                "delay_sla": delay_sla,
                "sla_violated": sla_violated,
            }
        )

    def collect(self) -> dict:
        return {
            "Instance ID": self.id,
            "Coordinates": self.coordinates,
            "Base Station": self.base_station.id if self.base_station else None,
        }


# ---------- resource management + stopping criterion ----------

def noop_resource_management(parameters: dict):
    # placeholder – all logic is in agent.step()
    return


def stopping_criterion(model: Simulator) -> bool:
    # stop when we reach max_steps
    return model.schedule.steps >= model.max_steps


# ---------- build infrastructure ----------

def build_infrastructure(sim: Simulator,
                         fog_coords=None,
                         wireless_delay=5.0,
                         link_delay=5.0,
                         link_bandwidth=10000):
    if fog_coords is None:
        fog_coords = [(0, 0), (250, 0), (500, 0), (750, 0)]

    # Topology
    topo = sim.initialize_agent(Topology())
    sim.topology = topo

    base_stations = []
    switches = []
    fog_servers = []

    # create BS + switches + fog servers
    for i, (x, y) in enumerate(fog_coords, start=1):
        bs = BaseStation()
        sim.initialize_agent(bs)
        bs.coordinates = [x, y]
        bs.wireless_delay = wireless_delay

        sw = NetworkSwitch()
        sim.initialize_agent(sw)
        bs._connect_to_network_switch(sw)

        # add to topology
        topo.add_node(sw)

        fog = FogServer(name=f"Fog-{i}")
        sim.initialize_agent(fog)
        bs._connect_to_edge_server(fog)

        base_stations.append(bs)
        switches.append(sw)
        fog_servers.append(fog)

    # cloud attached to first base station
    cloud = CloudServer()
    sim.initialize_agent(cloud)
    base_stations[0]._connect_to_edge_server(cloud)

    # linear links between switches
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

    return base_stations, switches, fog_servers, cloud


def create_users(sim: Simulator,
                 n_users=50,
                 area_width=800,
                 area_height=200,
                 data_size_min=100,
                 data_size_max=1000):
    users = []
    for i in range(1, n_users + 1):
        u = MovingUser(
            obj_id=i,
            area_width=area_width,
            area_height=area_height,
            data_size_min=data_size_min,
            data_size_max=data_size_max,
        )
        sim.initialize_agent(u)
        users.append(u)
    return users


# ---------- main experiment ----------

def run_experiment_markov_edgesimpy(
    n_users=50,
    max_steps=1000,
    area_width=800,
    area_height=200,
    data_size_min=100,
    data_size_max=1000,
    fog_coords=None,
    max_fog_distance=200.0,
    delay_sla=100.0,
    seed=0,
):
    random.seed(seed)

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
    sim.request_records = []

    base_stations, switches, fog_servers, cloud = build_infrastructure(
        sim,
        fog_coords=fog_coords,
        wireless_delay=5.0,
        link_delay=5.0,
        link_bandwidth=10000,
    )

    users = create_users(
        sim,
        n_users=n_users,
        area_width=area_width,
        area_height=area_height,
        data_size_min=data_size_min,
        data_size_max=data_size_max,
    )

    sim.run_model()

    # Build DataFrames from logs
    df_req = pd.DataFrame(sim.request_records)

    # Agent-level metrics from monitor()
    df_servers = pd.DataFrame(
        sim.agent_metrics.get("CloudServer", []) +
        sim.agent_metrics.get("FogServer", [])
    )

    df_users = pd.DataFrame(
        sim.agent_metrics.get("MovingUser", [])
    )

    return sim, df_req, df_servers, df_users


if __name__ == "__main__":
    sim, df_req, df_srv, df_usr = run_experiment_markov_edgesimpy()

    print("Total requests:", len(df_req))
    print(df_req.groupby("server_layer")["total_delay"].agg(["count", "mean", "std"]))

    df_req.to_csv("experiment_markov_fog_requests.csv", index=False)
    df_srv.to_csv("experiment_markov_fog_servers.csv", index=False)
    df_usr.to_csv("experiment_markov_fog_users.csv", index=False)
    print("Saved CSVs: requests, servers, users.")
