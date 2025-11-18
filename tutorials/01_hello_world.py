import random
from edge_sim_py.simulator import Simulator
from edge_sim_py.components import Topology, EdgeServer, User

# ------------------------------------------------------------
# 1️⃣ Define custom agents
# ------------------------------------------------------------

class MyEdgeServer(EdgeServer):
    def __init__(self, name, capacity=100):
        super().__init__()
        self.name = name
        self.capacity = capacity  # how many task units it can handle per step
        self.cpu_util = 0.0
        self.memory_util = 0.0
        self.energy = 0.0
        self.queue = []  # pending tasks

    def receive_task(self, task_size):
        self.queue.append(task_size)

    def step(self):
        # Process tasks based on available capacity
        total_load = sum(self.queue)
        processed = min(total_load, self.capacity)
        utilization = processed / self.capacity

        # Update CPU and memory realistically
        self.cpu_util = utilization * 100
        self.memory_util = 30 + utilization * 60  # baseline 30%
        self.energy += processed * 0.5  # energy proportional to work

        # Remove processed tasks
        self.queue = []

    def collect(self):
        return {
            "CPU (%)": round(self.cpu_util, 2),
            "Memory (%)": round(self.memory_util, 2),
            "Energy (J)": round(self.energy, 2),
            "Pending Tasks": len(self.queue),
        }


class MyUser(User):
    def __init__(self, name, server):
        super().__init__()
        self.name = name
        self.server = server
        self.requests_sent = 0

    def step(self):
        # Send 0–3 tasks of varying sizes each step
        num_tasks = random.randint(0, 3)
        for _ in range(num_tasks):
            task_size = random.randint(10, 40)
            self.server.receive_task(task_size)
            self.requests_sent += 1

    def collect(self):
        return {"Requests Sent": self.requests_sent}


# ------------------------------------------------------------
# 2️⃣ Simulation configuration
# ------------------------------------------------------------
def my_algorithm(parameters=None, **kwargs):
    if parameters is None:
        parameters = kwargs.get("parameters", {})
    step = parameters.get("current_step", 0)
    print(f"[Algorithm] Step {step} executed.")

def stopping_criterion(model):
    return model.schedule.steps == 5

sim = Simulator(
    stopping_criterion=stopping_criterion,
    resource_management_algorithm=my_algorithm,
    tick_duration=1,
    tick_unit="seconds",
    dump_interval=2,
    logs_directory="logs",
)

topology = sim.initialize_agent(Topology())
sim.topology = topology

server = sim.initialize_agent(MyEdgeServer("EdgeServer-1"))
user = sim.initialize_agent(MyUser("User-1", server))

sim.running = True
sim.run_model()

print("\n=== EDGE SERVER STATS ===")
for record in sim.agent_metrics.get("MyEdgeServer", []):
    print(record)

print("\n=== USER STATS ===")
for record in sim.agent_metrics.get("MyUser", []):
    print(record)
