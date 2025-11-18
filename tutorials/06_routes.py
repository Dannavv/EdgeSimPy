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
        latency = log_info["latency"] + random.randint(12,15)
        route = log_info["route"] + ["cloud"]
        # If cloud must forward to fog (for cloud+fog)
        if "forward_to_fog" in log_info:
            fog = log_info["forward_to_fog"]
            fog.receive_data(data_size, user, None, {
                "route": route,
                "latency": latency,
                "sensor_name": log_info.get("sensor_name","unknown"),
                "time": log_info.get("time",0),
                "direct": True})
        else:
            user.receive_data(data_size, route=route, latency=latency)
            user.request_log.append({
                "sensor": log_info.get("sensor_name","unknown"),
                "time": log_info.get("time",0),
                "route": route,
                "fulfilled": True,
                "latency": latency
            })
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
        return dist((self.x, self.y), point) <= self.radius
    def receive_data(self, data_size, user, sensor, log_info):
        self.request_load += data_size
        fog_latency = random.randint(2,4)
        curr_route = log_info["route"] + ["fog"]
        curr_latency = log_info["latency"] + fog_latency
        t = log_info["time"]
        # For direct delivery phase (cloud+fog capstone), no sensor is provided
        if sensor is None or log_info.get("direct", False):
            user.receive_data(data_size, route=curr_route, latency=curr_latency)
            user.request_log.append({
                "sensor": log_info.get("sensor_name", "unknown"),
                "time": t,
                "route": curr_route,
                "fulfilled": True,
                "latency": curr_latency
            })
            return
        # Standard coverage-based routing
        sensor_in_fog = self.in_coverage((sensor.x, sensor.y))
        user_in_fog = self.in_coverage((user.x, user.y))
        if sensor_in_fog and user_in_fog:
            user.receive_data(data_size, route=curr_route, latency=curr_latency)
            user.request_log.append({"sensor": sensor.name, "time": t, "route": curr_route, "fulfilled": True, "latency": curr_latency})
        elif sensor_in_fog and not user_in_fog:
            user.cloud.receive_data(data_size, user, {"route": curr_route, "latency": curr_latency, "sensor_name": sensor.name, "time": t})
        # cloud+fog handled from cloud's forward_to_fog (see below)
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
            "X": self.x, "Y": self.y, "Radius": self.radius
        }

def get_fog_covering(x, y, fogs):
    return [fog for fog in fogs if fog.in_coverage((x, y))]

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
        data_size = self.data_size
        latency = random.randint(1,2)
        sensor_fogs = get_fog_covering(self.x, self.y, self.fogs)
        user_fogs = get_fog_covering(user.x, user.y, self.fogs)
        # 1. fog + fog (both in same fog)
        for fog in sensor_fogs:
            if fog in user_fogs:
                fog.receive_data(data_size, user, self, {
                    "route": [], "latency": latency, "sensor_name": self.name, "time": t
                })
                return
        # 2. fog + cloud (sensor in fog, user out)
        if sensor_fogs:
            fog = sensor_fogs[0]
            fog.receive_data(data_size, user, self, {
                "route": [], "latency": latency, "sensor_name": self.name, "time": t
            })
            return
        # 3. cloud + fog (user in fog, sensor out)
        if user_fogs:
            user.cloud.receive_data(data_size, user, {
                "route": [], "latency": latency, "sensor_name": self.name, "time": t, "forward_to_fog": user_fogs[0]})
            return
        # 4. cloud + cloud
        user.cloud.receive_data(data_size, user, {
            "route": [], "latency": latency, "sensor_name": self.name, "time": t
        })
        user.request_log.append({"sensor": self.name, "time": t, "route": ["cloud", "cloud"], "fulfilled": True, "latency": latency + random.randint(12,15)})
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
        self.request_log = []
        self.latency_log = []
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

    for t in range(20):
        for user in users:
            user.current_time = t
        sim.step()
        if not sim.running:
            break
    sim.running = False

    print("\nRequest Fulfillment and Routes:")
    for user in users:
        total_req = len(user.request_log)
        fulfilled = sum(1 for log in user.request_log if log['fulfilled'])
        route_counts = {}
        for log in user.request_log:
            r_label = ' + '.join(log['route'])
            route_counts[r_label] = route_counts.get(r_label, 0) + 1
        avg_latency = sum(log['latency'] for log in user.request_log)/total_req if total_req else 0
        print(f"{user.name}: fulfilled/total={fulfilled}/{total_req}, routes={route_counts}, avg latency={avg_latency:.2f}ms")
    print("\nExample route log for first user:")
    print(users[0].request_log[:5])

if __name__ == "__main__":
    run_simulation()
