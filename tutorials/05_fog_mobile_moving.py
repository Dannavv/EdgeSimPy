import math
import random
import pandas as pd
from edge_sim_py.simulator import Simulator
from edge_sim_py.components import EdgeServer, User, Topology

def dist(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def random_walk(agent):
    agent.x += random.choice([-1, 0, 1])
    agent.y += random.choice([-1, 0, 1])

class CloudServer(EdgeServer):
    def __init__(self):
        super().__init__(None, None, "Cloud", 20000, 128000, 200000)
        self.x, self.y = 0, 0
        self.request_load = 0
        self.cpu_util = 0.0
        self.energy = 0.0
    def receive_data(self, data_size, user, log_info):
        self.request_load += data_size
        # Record that cloud served this request and add cloud latency
        user.receive_data(data_size, route=log_info["route"]+['cloud'], latency=log_info["latency"]+random.randint(12,15))
    def step(self):
        processed = min(self.request_load, self.cpu)
        utilization = processed / self.cpu if self.cpu > 0 else 0
        self.cpu_util = utilization * 100
        self.energy += processed * 0.3
        self.request_load = 0
    def collect(self):
        return {
            "CPU Util (%)": round(self.cpu_util, 2),
            "Energy (J)": round(self.energy, 2),
            "X": self.x, "Y": self.y,
        }

class FogNode(EdgeServer):
    def __init__(self, name, x, y, radius):
        super().__init__(None, None, name, 10000, 64000, 100000)
        self.x, self.y = x, y
        self.radius = radius
        self.request_load = 0
        self.cpu_util = 0.0
        self.energy = 0.0
        self.mobility_model = None
    def in_coverage(self, point):
        return dist((self.x,self.y), point) <= self.radius
    def receive_data(self, data_size, user, log_info):
        self.request_load += data_size
        # Increment latency for fog processing
        current_latency = log_info["latency"] + random.randint(2,4)
        if self.in_coverage((user.x, user.y)):
            user.receive_data(data_size, route=log_info["route"]+['fog'], latency=current_latency)
        else:
            user.cloud.receive_data(data_size, user, {"route":log_info["route"]+['fog'], "latency":current_latency})
    def step(self):
        if self.mobility_model is not None:
            self.mobility_model(self)
        processed = min(self.request_load, self.cpu)
        utilization = processed / self.cpu if self.cpu > 0 else 0
        self.cpu_util = utilization * 100
        self.energy += processed * 0.5
        self.request_load = 0
    def collect(self):
        return {
            "CPU Util (%)": round(self.cpu_util, 2),
            "Energy (J)": round(self.energy, 2),
            "X": self.x, "Y": self.y,
            "Radius": self.radius,
        }

class SensorDevice(User):
    def __init__(self, name, x, y, fogs, cloud):
        super().__init__()
        self.name = name
        self.x, self.y = x, y
        self.fogs = fogs
        self.cloud = cloud
        self.data_size = random.randint(10, 30)
        self.mobility_model = None
    def send_data(self, user, t):
        log_info = {"route":[], "latency":random.randint(1,2)}
        for fog in self.fogs:
            if fog.in_coverage((self.x, self.y)):
                fog.receive_data(self.data_size, user, log_info)
                user.request_log.append({"sensor": self.name, "time": t, "route": log_info["route"]+['fog'], "fulfilled": True})
                return
        self.cloud.receive_data(self.data_size, user, log_info)
        user.request_log.append({"sensor": self.name, "time": t, "route": log_info["route"]+['cloud'], "fulfilled": True})
    def collect(self):
        return {"X": self.x, "Y": self.y}
    def step(self):
        if self.mobility_model is not None:
            self.mobility_model(self)

class UserDevice(User):
    def __init__(self, name, x, y, sensors, cloud):
        super().__init__()
        self.name = name
        self.x, self.y = x, y
        self.sensors = sensors
        self.cloud = cloud
        self.received_data = 0
        self.mobility_model = None
        self.request_log = []  # [{sensor:..., time:..., route:..., fulfilled:...}]
        self.latency_log = []  # [latency values per request]
    def move(self):
        if self.mobility_model is not None:
            self.mobility_model(self)
    def request_data(self, t):
        sensor = random.choice(self.sensors)
        sensor.send_data(self, t)
    def receive_data(self, data_size, route, latency):
        self.received_data += data_size
        self.latency_log.append(latency)
    def step(self):
        self.move()
        # Probabilistically generate a request
        if random.random() > 0.5:
            self.request_data(getattr(self, "current_time", 0))
    def collect(self):
        return {"X": self.x, "Y": self.y, "Data Received": self.received_data}

def simple_rma(parameters=None, **kwargs):
    pass

def run_simulation():
    sim = Simulator(
        stopping_criterion=lambda model: model.schedule.steps >= 20,
        resource_management_algorithm=simple_rma,
        tick_duration=1,
        tick_unit="seconds",
        dump_interval=1,
        logs_directory=None,
    )
    topology = sim.initialize_agent(Topology())
    sim.topology = topology

    cloud = sim.initialize_agent(CloudServer())
    fog_static = sim.initialize_agent(FogNode("Fog_Static", 5, 5, 5))
    fog_moving1 = sim.initialize_agent(FogNode("Fog_Moving1", 10, 10, 4))
    fog_moving2 = sim.initialize_agent(FogNode("Fog_Moving2", -10, -10, 4))
    fogs = [fog_static, fog_moving1, fog_moving2]
    fog_moving1.mobility_model = random_walk
    fog_moving2.mobility_model = random_walk

    sensors_static = [
        sim.initialize_agent(SensorDevice(f"Sensor_Static_{i+1}", random.randint(-10,10), random.randint(-10,10), fogs, cloud))
        for i in range(3)
    ]
    sensors_mobile = [
        sim.initialize_agent(SensorDevice(f"Sensor_Mobile_{i+1}", 0, 0, fogs, cloud))
        for i in range(2)
    ]
    for sensor in sensors_mobile:
        sensor.mobility_model = random_walk
    sensors = sensors_static + sensors_mobile

    users_moving = [
        sim.initialize_agent(UserDevice(f"User_Mobile_{i+1}", 0, 0, sensors, cloud))
        for i in range(2)
    ]
    users_static = [
        sim.initialize_agent(UserDevice(f"User_Static_{i+1}", random.randint(-10,10), random.randint(-10,10), sensors, cloud))
        for i in range(2)
    ]
    for user in users_moving:
        user.mobility_model = random_walk
    users = users_moving + users_static

    # Track simulated time for logging in users
    for t in range(20):
        for user in users:
            user.current_time = t
        sim.step()  # manually step the simulation
        if not sim.running:
            break
    sim.running = False
    # Print summary statistics per user
    print("\nRequest Fulfillment and Routes:")
    for user in users:
        total_req = len(user.request_log)
        fulfilled = sum(1 for log in user.request_log if log['fulfilled'])
        routes = [tuple(log['route']) for log in user.request_log]
        fog_only = sum(1 for r in routes if r==('fog',))
        fog_cloud = sum(1 for r in routes if r==('fog','cloud'))
        cloud_only = sum(1 for r in routes if r==('cloud',))
        avg_latency = sum(user.latency_log)/len(user.latency_log) if user.latency_log else 0
        print(f"{user.name}: fulfilled/total={fulfilled}/{total_req}, fog_only={fog_only}, fog+cloud={fog_cloud}, cloud_only={cloud_only}, avg latency={avg_latency:.2f}ms")

    print("\nExample route log for first user:")
    print(users[0].request_log[:5])
    print("\nFull data per user is in request_log and latency_log attributes.")

if __name__ == "__main__":
    run_simulation()
