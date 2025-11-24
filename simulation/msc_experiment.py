# ======================================================================================
# Filename: msc_experiment.py
# Description: Implements a simulation experiment for the Mobile Sensor-Cloud (MSC)
#              architecture, utilizing core components from the EdgeSimPy framework.
# ======================================================================================

# EdgeSimPy Component Imports
from edge_sim_py.simulator import Simulator
from edge_sim_py.components.application import Application
from edge_sim_py.components.base_station import BaseStation
from edge_sim_py.components.edge_server import EdgeServer
from edge_sim_py.components.network_link import NetworkLink
from edge_sim_py.components.network_switch import NetworkSwitch
from edge_sim_py.components.service import Service
from edge_sim_py.components.topology import Topology
from edge_sim_py.components.user import User
from edge_sim_py.components.mobility_models.random_mobility import random_mobility

# Python Standard Library Imports
import numpy as np
from typing import List
from math import sqrt

# --- MSC Constants ---
MAX_VELOCITY = 100         
INITIAL_ENERGY = 100       
THRESHOLD_ENERGY = 25      
THRESHOLD_ENERGY_RATIO = THRESHOLD_ENERGY / INITIAL_ENERGY
GAMMA = 0.5                


# ======================================================================================
# 1. CUSTOM MSC COMPONENTS
# ======================================================================================

def euclidean_distance(c1: tuple, c2: tuple) -> float:
    """Calculates Euclidean distance for d_i(tau)"""
    return sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


class MobileSensorHost(User):
    """
    Extends the EdgeSimPy User class to represent a Mobile Device host carrying a Physical Sensor Node.
    """
    def __init__(
        self, 
        initial_energy=INITIAL_ENERGY, 
        energy_utilized=0, 
        sensing_radius=30, 
        estimated_sop=0, 
        *args, 
        **kwargs
    ):
        # Capture custom attributes
        self._initial_energy = initial_energy
        self._energy_utilized = energy_utilized
        self._sensing_radius = sensing_radius
        self._estimated_sop = estimated_sop
        
        # Call base constructor
        super().__init__(*args, **kwargs) 

        # Set custom attributes
        self.initial_energy = self._initial_energy
        self.energy_utilized = self._energy_utilized
        self.sensing_radius = self._sensing_radius
        self.estimated_sop = self._estimated_sop
        
        # Initialize other MSC properties
        self.current_velocity = 0
        self.time_present_in_app_area = 0 
        self.total_monitored_time = 0 
        
        # Explicitly initialize the services list
        self.services = [] 
        
    def step(self):
        """Overrides User.step() to ensure mobility and energy updates occur."""
        # 1. Execute mobility
        if hasattr(self, "mobility_model") and self.mobility_model is not None:
            self.mobility_model(user=self)

        # 2. Update velocity & energy
        self.current_velocity = np.random.uniform(0, MAX_VELOCITY)
        self.energy_utilized += 1 


class VirtualSensor(Service):
    """Extends the Service class to represent a Virtual Sensor (VS_i)."""
    def __init__(self, *args, min_sensor_count=3, **kwargs):
        self._min_sensor_count = min_sensor_count 
        super().__init__(*args, **kwargs)
        self.min_sensor_count = self._min_sensor_count
        

# ======================================================================================
# 2. MSC UTILITY FUNCTIONS
# ======================================================================================

def calculate_msc_fitness(sensor: MobileSensorHost, grid_point_coords: tuple) -> float:
    """Implements the Fitness Function F_s_i(tau_j) from Equation (17)."""
    
    # Constraint Checks
    norm_res_energy = (sensor.initial_energy - sensor.energy_utilized) / sensor.initial_energy
    
    if norm_res_energy < THRESHOLD_ENERGY_RATIO:
        return -float('inf') 

    dist = euclidean_distance(sensor.coordinates, grid_point_coords)
    if dist > sensor.sensing_radius:
        return -float('inf') 

    # Parameter Calculation
    sensing_radius = sensor.sensing_radius       
    norm_velocity = sensor.current_velocity / MAX_VELOCITY 
    sop = sensor.estimated_sop                   
    
    # Fitness Function
    term_A = sop - (dist / sensing_radius) - norm_velocity
    fitness = term_A * norm_res_energy
    
    return fitness


def msc_reallocation_policy(parameters: dict):
    """Executes the SCSP's core decision-making for Sensor Reallocation."""
    model: Simulator = parameters["model"]
    current_step = model.schedule.steps + 1
    
    virtual_sensors: List[VirtualSensor] = VirtualSensor.all()
    
    for vs in virtual_sensors:
        
        # Check if reallocation is needed
        if len(vs.users) < vs.min_sensor_count:
            
            target_location = vs.server.coordinates
            
            # Find available hosts (not currently in this VS)
            available_hosts = [
                h for h in MobileSensorHost.all() 
                if h not in vs.users and h.base_station is not None
            ]
            
            # Find best sensor (Algorithm 1)
            best_sensor = None
            max_fitness = -float('inf')
            
            for sensor in available_hosts:
                # Mock SoP update
                current_presence_ratio = np.random.uniform(0.1, 1.0) 
                sensor.estimated_sop = (GAMMA * current_presence_ratio) + ((1 - GAMMA) * sensor.estimated_sop)
                
                fitness = calculate_msc_fitness(sensor, target_location)
                
                if fitness > max_fitness:
                    max_fitness = fitness
                    best_sensor = sensor
            
            # Reallocate
            if best_sensor and max_fitness > -float('inf'):
                vs.users.append(best_sensor)
                best_sensor.services.append(vs)
                
                if vs.application and best_sensor not in vs.application.users:
                    vs.application.users.append(best_sensor)
                
                print(f"[{current_step}] REALLOCATED: {best_sensor} (F: {max_fitness:.2f}) to {vs}. Total: {len(vs.users)}/{vs.min_sensor_count}")
            
            elif len(vs.users) < vs.min_sensor_count:
                print(f"[{current_step}] FAILED REALLOCATION: No suitable sensor found for {vs}.")
                

# ======================================================================================
# 3. SIMULATION SETUP
# ======================================================================================

if __name__ == "__main__":
    
    # --- Infrastructure Setup ---
    msc_simulator = Simulator(
        resource_management_algorithm=msc_reallocation_policy,
        user_defined_functions=[random_mobility], 
        tick_duration=1,
        tick_unit="minutes",
        dump_interval=float('inf')
    )
    
    # --- FIX: Inject 'model' into parameters manually ---
    msc_simulator.resource_management_algorithm_parameters["model"] = msc_simulator

    def stopping_criterion(model: Simulator):
        return model.schedule.steps >= 5

    msc_simulator.stopping_criterion = stopping_criterion
    
    # 1. Topology & Network
    topology = Topology()
    msc_simulator.initialize_agent(topology)

    bs1 = BaseStation()
    msc_simulator.initialize_agent(bs1)
    bs1.coordinates = (100, 100) 
    
    switch1 = NetworkSwitch()
    msc_simulator.initialize_agent(switch1)
    bs1.network_switch = switch1
    
    edge_server = EdgeServer(cpu=1000, memory=1000, disk=1000)
    msc_simulator.initialize_agent(edge_server)
    edge_server.coordinates = (100, 100)
    edge_server.base_station = bs1

    # 2. Application and Virtual Sensor
    app = Application(label="MSC_Monitoring_App")
    msc_simulator.initialize_agent(app)

    vs_service = VirtualSensor(label="VirtualSensor_Temp", min_sensor_count=2, cpu_demand=10, memory_demand=10)
    msc_simulator.initialize_agent(vs_service)
    vs_service.server = edge_server
    vs_service.application = app

    # 3. Mobile Sensor Hosts
    
    # Sensor 1
    sensor1 = MobileSensorHost(sensing_radius=30, estimated_sop=0.9, energy_utilized=0)
    msc_simulator.initialize_agent(sensor1)
    sensor1.coordinates = (110, 110) 
    sensor1.memory_demand = 1
    sensor1.cpu_demand = 1
    sensor1.base_station = bs1
    sensor1.mobility_model = random_mobility 

    # Sensor 2
    sensor2 = MobileSensorHost(sensing_radius=30, estimated_sop=0.7, energy_utilized=50) 
    msc_simulator.initialize_agent(sensor2)
    sensor2.coordinates = (105, 105) 
    sensor2.memory_demand = 1
    sensor2.cpu_demand = 1
    sensor2.base_station = bs1
    sensor2.mobility_model = random_mobility
    
    # Sensor 3
    sensor3 = MobileSensorHost(sensing_radius=30, estimated_sop=0.1, energy_utilized=5) 
    msc_simulator.initialize_agent(sensor3)
    sensor3.coordinates = (500, 500) 
    sensor3.memory_demand = 1
    sensor3.cpu_demand = 1
    sensor3.base_station = bs1
    sensor3.mobility_model = random_mobility

    # Set Initial State
    vs_service.users.append(sensor2)
    sensor2.services.append(vs_service) 
    app.users.append(sensor2)
    
    # --- Run ---
    print("=====================================================")
    print("  Starting Mobile Sensor-Cloud (MSC) Simulation      ")
    print("=====================================================")
    print(f"Target VS: {vs_service.label} requires {vs_service.min_sensor_count} active sensors.")
    print(f"Initial Active Sensors: {[str(s) for s in vs_service.users]}\n")
    
    msc_simulator.run_model()
    
    print("\n=====================================================")
    print("            MSC Simulation Finished                ")
    print(f"Final Active Sensors: {[str(s) for s in vs_service.users]}")
    print("=====================================================")