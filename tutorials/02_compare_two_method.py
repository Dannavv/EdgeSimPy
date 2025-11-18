# ============================================================
# ✅ Compare Two Methods in Fog Simulation (FINAL FIXED & CLEAN)
# ============================================================

import random
import pandas as pd
from edge_sim_py.simulator import Simulator
from edge_sim_py.components import Topology, EdgeServer, User


# ------------------------------------------------------------
# 1️⃣ Define Agents
# ------------------------------------------------------------
class MyEdgeServer(EdgeServer):
    def __init__(self, name, capacity=100):
        super().__init__()
        self.name = name
        self.capacity = capacity
        self.cpu_util = 0.0
        self.memory_util = 0.0
        self.energy = 0.0
        self.queue = []

    def receive_task(self, task_size):
        self.queue.append(task_size)

    def step(self):
        total_load = sum(self.queue)
        processed = min(total_load, self.capacity)
        utilization = processed / self.capacity if self.capacity > 0 else 0

        self.cpu_util = utilization * 100
        self.memory_util = 30 + utilization * 60
        self.energy += processed * 0.5  # simple linear model

        # clear queue for next tick
        self.queue = []

    def collect(self):
        return {
            "CPU (%)": round(self.cpu_util, 2),
            "Memory (%)": round(self.memory_util, 2),
            "Energy (J)": round(self.energy, 2),
        }


class MyUser(User):
    def __init__(self, name, server):
        super().__init__()
        self.name = name
        self.server = server
        self.requests_sent = 0

    def step(self):
        num_tasks = random.randint(0, 3)
        for _ in range(num_tasks):
            task_size = random.randint(10, 40)
            self.server.receive_task(task_size)
            self.requests_sent += 1

    def collect(self):
        return {"Requests Sent": self.requests_sent}


# ------------------------------------------------------------
# 2️⃣ Define Two Methods (Algorithms)
# ------------------------------------------------------------
def method_a(parameters=None, **kwargs):
    step = parameters.get("current_step", 0) if parameters else 0
    for agent in MyEdgeServer.all():
        agent.capacity = 100
    print(f"[Method A] Step {step} executed (capacity=100).")

def method_b(parameters=None, **kwargs):
    step = parameters.get("current_step", 0) if parameters else 0
    for agent in MyEdgeServer.all():
        agent.capacity = 60
    print(f"[Method B] Step {step} executed (capacity=60).")



# ------------------------------------------------------------
# 3️⃣ Common setup function
# ------------------------------------------------------------
def run_simulation(algorithm_fn, label):
    def stopping_criterion(model):
        return model.schedule.steps >= 5  # ✅ safer stop condition

    sim = Simulator(
        stopping_criterion=stopping_criterion,
        resource_management_algorithm=algorithm_fn,
        tick_duration=1,
        tick_unit="seconds",
        dump_interval=1,
        logs_directory=None,
    )

    topology = sim.initialize_agent(Topology())
    sim.topology = topology

    server = sim.initialize_agent(MyEdgeServer(f"{label}_EdgeServer"))
    user = sim.initialize_agent(MyUser(f"{label}_User", server))

    sim.running = True
    sim.run_model()

    # ✅ FIX: use correct class key in agent_metrics
    server_data = pd.DataFrame(sim.agent_metrics.get("MyEdgeServer", []))
    user_data = pd.DataFrame(sim.agent_metrics.get("MyUser", []))

    # sometimes metrics are stored under actual class name
    if server_data.empty:
        for k in sim.agent_metrics.keys():
            if "EdgeServer" in k:
                server_data = pd.DataFrame(sim.agent_metrics[k])
    if user_data.empty:
        for k in sim.agent_metrics.keys():
            if "User" in k:
                user_data = pd.DataFrame(sim.agent_metrics[k])

    return server_data, user_data


# ------------------------------------------------------------
# 4️⃣ Run both simulations
# ------------------------------------------------------------
server_a, user_a = run_simulation(method_a, "MethodA")
server_b, user_b = run_simulation(method_b, "MethodB")


# ------------------------------------------------------------
# 5️⃣ Compare results
# ------------------------------------------------------------
summary = pd.DataFrame({
    "Method": ["A", "B"],
    "Avg_CPU": [server_a["CPU (%)"].mean(), server_b["CPU (%)"].mean()],
    "Total_Energy": [server_a["Energy (J)"].iloc[-1], server_b["Energy (J)"].iloc[-1]],
    "Total_Requests": [user_a["Requests Sent"].iloc[-1], user_b["Requests Sent"].iloc[-1]],
})

print("\n=== 📊 Comparison Summary ===")
print(summary)
