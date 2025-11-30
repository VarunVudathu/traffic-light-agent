"""
Baseline controllers for comparison.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "CityFlow" / "build"))
import cityflow
import numpy as np

# Import MaxPressure utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.maxpressure import compute_maxpressure_reward, estimate_lane_pressure_by_phase


class FixedTimeBaseline:
    """Fixed-time traffic light controller (cycles through phases)."""

    def __init__(self, config_path, phase_duration=30, max_steps=1000):
        self.eng = cityflow.Engine(config_path, thread_num=1)
        self.intersection_ids = self.eng.get_intersection_ids()
        self.phase_duration = phase_duration
        self.max_steps = max_steps

        # Track state for each intersection
        self.intersection_state = {
            iid: {'current_phase': 0, 'time_in_phase': 0}
            for iid in self.intersection_ids
        }

    def run(self):
        """Run the baseline controller."""
        self.eng.reset()
        rewards = []

        for step in range(self.max_steps):
            # Update traffic lights
            for iid, state in self.intersection_state.items():
                phases = self.eng.get_intersection_phase(iid)
                if not phases:
                    continue

                state['time_in_phase'] += 1

                # Switch phase after duration
                if state['time_in_phase'] >= self.phase_duration:
                    state['current_phase'] = (state['current_phase'] + 1) % len(phases)
                    state['time_in_phase'] = 0

                self.eng.set_tl_phase(iid, state['current_phase'])

            # Step simulation
            self.eng.next_step()

            # Compute reward
            waiting_counts = list(self.eng.get_lane_waiting_vehicle_count().values())
            reward = -np.mean(waiting_counts) if waiting_counts else 0.0
            rewards.append(reward)

        return rewards


class MaxPressureBaseline:
    """MaxPressure heuristic controller (selects phase with max pressure)."""

    def __init__(self, config_path, min_phase_duration=5, max_steps=1000):
        self.eng = cityflow.Engine(config_path, thread_num=1)
        self.intersection_ids = self.eng.get_intersection_ids()
        self.min_phase_duration = min_phase_duration
        self.max_steps = max_steps

        # Track state for each intersection
        self.intersection_state = {
            iid: {'current_phase': 0, 'time_in_phase': 0, 'num_phases': 0}
            for iid in self.intersection_ids
        }

        # Get number of phases per intersection
        for iid in self.intersection_ids:
            phases = self.eng.get_intersection_phase(iid)
            self.intersection_state[iid]['num_phases'] = len(phases) if phases else 4

    def run(self):
        """Run the MaxPressure controller."""
        self.eng.reset()
        rewards = []

        for step in range(self.max_steps):
            # Update traffic lights using MaxPressure heuristic
            for iid, state in self.intersection_state.items():
                state['time_in_phase'] += 1

                # Only switch if minimum duration met
                if state['time_in_phase'] >= self.min_phase_duration:
                    # Compute pressure for each phase
                    num_phases = state['num_phases']
                    pressures = estimate_lane_pressure_by_phase(
                        self.eng, iid, num_phases
                    )

                    # Select phase with maximum pressure
                    best_phase = int(np.argmax(pressures))

                    # Switch if different from current
                    if best_phase != state['current_phase']:
                        state['current_phase'] = best_phase
                        state['time_in_phase'] = 0

                self.eng.set_tl_phase(iid, state['current_phase'])

            # Step simulation
            self.eng.next_step()

            # Compute reward using MaxPressure
            reward = compute_maxpressure_reward(
                self.eng,
                self.intersection_ids[0] if self.intersection_ids else None,
                self.intersection_state[self.intersection_ids[0]]['current_phase'] if self.intersection_ids else 0
            )
            rewards.append(reward)

        return rewards
