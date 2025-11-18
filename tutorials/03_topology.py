import random
import pandas as pd
from edge_sim_py.simulator import Simulator
from edge_sim_py.components import Topology, EdgeServer, User

# ------------------------------
# Custom agent classes
# ------------------------------

class CloudServer(EdgeServer):
    def __init__(self, name):
        super().__init__(None, None, name, 10000, 64000, 100000)
        self.name = name

class FogNode(EdgeServer):
    def __init__(self, name):
        super().__init__(None, None, name, 1000, 16000, 20000)
        self.name = name
        self.cpudemand = 0  
        self.cpu_util = 0.0
        self.energy = 0.0

    def step(self):
        # Process CPU demand with capacity limit
        processed = min(self.cpudemand, self.cpu)
        utilization = processed / self.cpu if self.cpu > 0 else 0
        self.cpu_util = utilization * 100
        self.energy += processed * 0.5  # Energy model (Joules)
        self.cpudemand = 0  # Reset demand after processing

    def collect(self):
        return {
            "CPU Util (%)": round(self.cpu_util, 2),
            "Energy (J)": round(self.energy, 2),
        }

class SensorDevice(User):
    def __init__(self, name, fog):
        super().__init__()
        self.name = name
        self.fog = fog
        self.requests_sent = 0

    def step(self):
        if random.random() > 0.7:
            task_size = random.randint(5, 15)
            self.fog.cpudemand += task_size
            self.requests_sent += 1

    def collect(self):
        return {"Requests Sent": self.requests_sent}

# ------------------------------
# Simple resource management algorithm
# ------------------------------

def simple_rma(parameters=None, **kwargs):
    # Could add dynamic resource adjustments here; empty for now
    pass

# ------------------------------
# Simulation function
# ------------------------------

def run_cloud_fog_iot():
    sim = Simulator(
        stopping_criterion=lambda model: model.schedule.steps >= 5,
        resource_management_algorithm=simple_rma,
        tick_duration=1,
        tick_unit="seconds",
        dump_interval=1,
        logs_directory=None,
    )

    topology = sim.initialize_agent(Topology())
    sim.topology = topology

    cloud = sim.initialize_agent(CloudServer("Cloud"))

    fogs = [sim.initialize_agent(FogNode(f"FogNode_{i+1}")) for i in range(3)]

    sensors = []
    for fi, fog in enumerate(fogs):
        devices = [sim.initialize_agent(SensorDevice(f"SensorDevice_{fi*5 + i + 1}", fog)) for i in range(5)]
        sensors.extend(devices)

    sim.running = True
    sim.run_model()

    # Metrics aggregation

    cloud_metrics = pd.DataFrame()
    for k in sim.agent_metrics:
        if "CloudServer" in k:
            cloud_metrics = pd.DataFrame(sim.agent_metrics[k])
    fog_metrics = pd.DataFrame()
    for k in sim.agent_metrics:
        if "FogNode" in k:
            fog_metrics = pd.DataFrame(sim.agent_metrics[k])
    sensor_metrics = pd.DataFrame()
    for k in sim.agent_metrics:
        if "SensorDevice" in k:
            sensor_metrics = pd.DataFrame(sim.agent_metrics[k])

    print("\nCloud Metrics:\n", cloud_metrics)
    print("\nFog Node Metrics:\n", fog_metrics)
    print("\nSensor Device Metrics:\n", sensor_metrics)

if __name__ == "__main__":
    run_cloud_fog_iot()
