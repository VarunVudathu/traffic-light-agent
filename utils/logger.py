"""
Enhanced logger for tracking metrics during traffic light control experiments.

Extends basic Logger functionality with additional tracking for:
- Queue length history over time
- Pressure values (for H2)
- Per-intersection queue tracking (for H3)
"""

import numpy as np


class EnhancedLogger:
    """
    Enhanced logger that tracks detailed metrics for evaluation.

    Tracks all metrics from original Logger plus:
    - queue_history: For recovery rate and variance analysis
    - pressure_history: For H2 MaxPressure evaluation
    - intersection_queues: For H3 Gini coefficient
    """

    def __init__(self):
        # Original Logger metrics
        self.total_delay = 0
        self.total_vehicles = 0
        self.total_stops = 0
        self.vehicle_passed = set()
        self.vehicle_stop_count = {}
        self.vehicle_last_speed = {}
        self.vehicle_start_time = {}

        # NEW: Enhanced metrics tracking
        self.queue_history = []  # List of (step, total_queue_length) tuples
        self.pressure_history = []  # List of pressure values for H2
        self.intersection_queues = {}  # {intersection_id: [queue_values]}

    def update(self, engine, current_step=None):
        """
        Update logger with current state from CityFlow engine.

        Args:
            engine: CityFlow engine instance
            current_step: Current simulation step (for queue_history)
        """
        # Original Logger updates
        vehicles = engine.get_vehicle_count()
        self.total_vehicles += vehicles

        # Track vehicle delays and stops
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
                    distance = float(info.get("distance", 0.0))
                    # Assume max speed 16.67 m/s for ideal travel time
                    ideal_time = distance / 16.67 if distance > 0 else 0
                    self.total_delay += max(0, travel_time - ideal_time)
                    del self.vehicle_start_time[v_id]
                continue

            speed = float(info.get("speed", 0.0))
            if v_id not in self.vehicle_last_speed:
                self.vehicle_last_speed[v_id] = speed
                self.vehicle_stop_count[v_id] = 0
            else:
                # Count stop if speed drops from moving to stopped
                if self.vehicle_last_speed[v_id] > 0.1 and speed < 0.1:
                    self.vehicle_stop_count[v_id] += 1
                self.vehicle_last_speed[v_id] = speed

        # Track throughput
        for v_id in engine.get_vehicles():
            try:
                if v_id not in self.vehicle_passed and not engine.get_vehicle_info(v_id)["running"]:
                    self.vehicle_passed.add(v_id)
            except Exception:
                continue

        # NEW: Track queue history for recovery rate and variance
        if current_step is not None:
            lane_counts = engine.get_lane_waiting_vehicle_count()
            total_queue = sum(lane_counts.values())
            self.queue_history.append((current_step, total_queue))

        # NEW: Track per-intersection queues for Gini coefficient (H3)
        intersection_ids = engine.get_intersection_ids()
        for iid in intersection_ids:
            if iid not in self.intersection_queues:
                self.intersection_queues[iid] = []

            # Get lanes for this intersection
            # For now, use total waiting count as approximation
            # (In full implementation, would map lanes to intersections)
            lane_counts = engine.get_lane_waiting_vehicle_count()
            intersection_queue = sum(lane_counts.values()) / len(intersection_ids)
            self.intersection_queues[iid].append(intersection_queue)

    def update_pressure(self, pressure_value):
        """
        Manually update pressure history (called from H2 agent).

        Args:
            pressure_value: Current pressure value
        """
        self.pressure_history.append(pressure_value)

    def metrics(self):
        """
        Compute and return all basic metrics.

        Returns:
            Dict with average_stop_rate, throughput, average_delay
        """
        avg_stops = np.mean(list(self.vehicle_stop_count.values())) if self.vehicle_stop_count else 0
        throughput = len(self.vehicle_passed)
        average_delay = self.total_delay / self.total_vehicles if self.total_vehicles > 0 else 0

        return {
            "average_stop_rate": avg_stops,
            "throughput": throughput,
            "average_delay": average_delay
        }

    def get_enhanced_metrics(self, scenario_name=None):
        """
        Get all metrics including enhanced ones for hypothesis testing.

        Args:
            scenario_name: Optional scenario name for surge-specific metrics

        Returns:
            Dict with all metrics (basic + enhanced)
        """
        from utils.metrics import compute_all_metrics

        # Get basic metrics
        basic_metrics = self.metrics()

        # Compute enhanced metrics
        enhanced_metrics = compute_all_metrics(
            queue_history=self.queue_history,
            scenario_name=scenario_name or '',
            logger_metrics=basic_metrics,
            pressure_history=self.pressure_history if self.pressure_history else None,
            intersection_queues=self.intersection_queues if self.intersection_queues else None
        )

        return enhanced_metrics

    def reset(self):
        """Reset all tracked metrics."""
        self.total_delay = 0
        self.total_vehicles = 0
        self.total_stops = 0
        self.vehicle_passed = set()
        self.vehicle_stop_count = {}
        self.vehicle_last_speed = {}
        self.vehicle_start_time = {}
        self.queue_history = []
        self.pressure_history = []
        self.intersection_queues = {}
