import cityflow
import numpy as np
import time
import os

# python generate_grid_scenario.py 2 3 --roadnetFile roadnet.json --flowFile flow.json --dir . --tlPlan

CONFIG = {
    "config_file": "data/config.json",
    "roadnet_file": "data/roadnet.json",
    "flow_file": "data/flow.json",
    "interval": 1,
    "steps": 1000
}

class Logger:
    def __init__(self):
        self.total_delay = 0
        self.total_vehicles = 0
        self.total_stops = 0
        self.vehicle_passed = set()
        self.vehicle_stop_count = {}
        self.vehicle_last_speed = {}
        self.vehicle_start_time = {}

    def update(self, engine):
        vehicles = engine.get_vehicle_count()
        self.total_vehicles += vehicles

        for v_id in engine.get_vehicles():
            if v_id not in self.vehicle_start_time:
                self.vehicle_start_time[v_id] = engine.get_current_time()

            try:
                info = engine.get_vehicle_info(v_id)
            except Exception:
                continue

            if not info.get("running", False):
                if v_id in self.vehicle_start_time:
                    travel_time = engine.get_current_time() - self.vehicle_start_time[v_id]
                    self.total_delay += travel_time - (float(info.get("distance", 0.0)) / 16.67)
                    del self.vehicle_start_time[v_id]
                continue

            speed = float(info.get("speed", 0.0))
            if v_id not in self.vehicle_last_speed:
                self.vehicle_last_speed[v_id] = speed
                self.vehicle_stop_count[v_id] = 0
            else:
                if self.vehicle_last_speed[v_id] > 0.1 and speed < 0.1:
                    self.vehicle_stop_count[v_id] += 1
                self.vehicle_last_speed[v_id] = speed

        for v_id in engine.get_vehicles():
            if v_id not in self.vehicle_passed and not engine.get_vehicle_info(v_id)["running"]:
                self.vehicle_passed.add(v_id)

    def metrics(self):
        avg_stops = np.mean(list(self.vehicle_stop_count.values())) if self.vehicle_stop_count else 0
        throughput = len(self.vehicle_passed)
        average_delay = self.total_delay / self.total_vehicles if self.total_vehicles else 0
        return {
            "average_stop_rate": avg_stops,
            "throughput": throughput,
            "average_delay": average_delay
        }

class BaseAgent:
    def __init__(self):
        self.env = cityflow.Engine(CONFIG["config_file"], thread_num=1)
        self.interval = CONFIG["interval"]
        self.steps = CONFIG["steps"]
        self.logger = Logger()

    def act(self, time_step):
        pass

    def run(self):
        rewards = []
        for t in range(self.steps):
            self.act(t)
            self.env.next_step()
            rewards.append(self.get_reward())
            self.logger.update(self.env)
        return rewards, self.logger.metrics()

    def get_reward(self):
        return -np.mean(list(self.env.get_lane_waiting_vehicle_count().values()))

class FixedLightBaseline(BaseAgent):
    def __init__(self):
        super().__init__()
        self.intersection_state = {
            inter_id: {'current_phase': 0, 'time_in_phase': 0}
            for inter_id in self.env.get_intersection_ids()
        }
        self.phase_duration = 30  # seconds

    def act(self, time_step):
        for inter_id, state in self.intersection_state.items():
            phases = self.env.get_intersection_phase(inter_id)
            if not phases:
                continue

            state['time_in_phase'] += 1
            if state['time_in_phase'] >= self.phase_duration:
                state['current_phase'] = (state['current_phase'] + 1) % len(phases)
                state['time_in_phase'] = 0

            self.env.set_tl_phase(inter_id, state['current_phase'])

class SingleAgent(BaseAgent):
    def act(self, time_step):
        for inter_id in self.env.get_intersection_ids():
            phases = self.env.get_intersection_phase(inter_id)
            best_phase = 0
            max_wait = -1
            for phase_id in range(len(phases)):
                self.env.set_tl_phase(inter_id, phase_id)
                for _ in range(10):
                    self.env.next_step()
                wait = -np.mean(list(self.env.get_lane_waiting_vehicle_count().values()))
                if wait > max_wait:
                    best_phase = phase_id
                    max_wait = wait
            self.env.set_tl_phase(inter_id, best_phase)

class EnhancedSingleAgent(SingleAgent):
    def get_reward(self):
        return -self.env.get_average_travel_time()

class MultiAgent(BaseAgent):
    def act(self, time_step):
        for inter_id in self.env.get_intersection_ids():
            phases = self.env.get_intersection_phase(inter_id)
            best_phase = 0
            max_clear = -1
            for phase_id in range(len(phases)):
                self.env.set_tl_phase(inter_id, phase_id)
                for _ in range(10):
                    self.env.next_step()
                lane_counts = self.env.get_lane_vehicle_count()
                clear = -np.mean(list(lane_counts.values()))
                if clear > max_clear:
                    best_phase = phase_id
                    max_clear = clear
            self.env.set_tl_phase(inter_id, best_phase)