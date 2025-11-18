import random
import pandas as pd
from edge_sim_py.simulator import Simulator
from edge_sim_py.components import EdgeServer, User, Topology

# ------------------------------
# Custom Classes
# ------------------------------
import math
import random

class CloudServer(EdgeServer):
    def __init__(self, name, x=0, y=0):
        super().__init__(None, None, name, 10000, 64000, 100000)
        self.name = name
        self.x = x
        self.y = y
        self.request_load = 0
        self.cpu_util = 0.0
        self.energy = 0.0

    def receive_request(self, load):
        self.request_load += load

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
            "X": self.x,
            "Y": self.y,
        }

class MobileDevice(User):
    CONNECTION_RANGE = 1  # maximum distance to maintain connection

    def __init__(self, name, cloud, x=0, y=0):
        super().__init__()
        self.name = name
        self.cloud = cloud
        self.x = x
        self.y = y
        self.requests_sent = 0

    def distance_to_cloud(self):
        dx = self.x - self.cloud.x
        dy = self.y - self.cloud.y
        return math.sqrt(dx*dx + dy*dy)

    def is_connected(self):
        return self.distance_to_cloud() <= self.CONNECTION_RANGE

    def step(self):
        # Move randomly
        self.x += random.choice([-1, 0, 1])
        self.y += random.choice([-1, 0, 1])

        if self.is_connected():
            # Generate and send task only if connected
            if random.random() > 0.5:
                task_size = random.randint(5, 15)
                self.requests_sent += 1
                self.cloud.receive_request(task_size)
        else:
            # If disconnected, no sending; perhaps queue tasks or wait

            # Optional: attempt moving towards cloud to reconnect
            # dx = self.cloud.x - self.x
            # dy = self.cloud.y - self.y
            # step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
            # step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
            # self.x += step_x
            # self.y += step_y
            pass

    def collect(self):
        return {
            "Requests Sent": self.requests_sent,
            "X Position": self.x,
            "Y Position": self.y,
            "Connected": self.is_connected(),
        }


# ------------------------------
# Simple resource management algorithm
# ------------------------------

def simple_rma(parameters=None, **kwargs):
    # Extend for resource management if needed
    pass

# ------------------------------
# Simulation function
# ------------------------------

def run_cloud_mobile_simulation():
    sim = Simulator(
        stopping_criterion=lambda model: model.schedule.steps >= 10,  # run 10 steps
        resource_management_algorithm=simple_rma,
        tick_duration=1,
        tick_unit="seconds",
        dump_interval=1,
        logs_directory=None,
    )
    topology = sim.initialize_agent(Topology())
    sim.topology = topology

    cloud = sim.initialize_agent(CloudServer("Cloud"))

    devices = [sim.initialize_agent(MobileDevice(f"MobileDevice_{i+1}", cloud, x=0, y=0)) for i in range(10)]

    sim.running = True
    sim.run_model()

    cloud_metrics = pd.DataFrame()
    for k in sim.agent_metrics:
        if "CloudServer" in k:
            cloud_metrics = pd.DataFrame(sim.agent_metrics[k])
    mobile_metrics = pd.DataFrame()
    for k in sim.agent_metrics:
        if "MobileDevice" in k:
            mobile_metrics = pd.DataFrame(sim.agent_metrics[k])

    print("\nCloud Metrics:\n", cloud_metrics)
    print("\nMobile Devices Metrics:\n", mobile_metrics)

if __name__ == "__main__":
    run_cloud_mobile_simulation()
