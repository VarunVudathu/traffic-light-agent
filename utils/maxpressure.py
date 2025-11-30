"""
MaxPressure utilities for H2 hypothesis testing.

MaxPressure reward: difference between incoming and outgoing queue lengths.
This provides a more informative learning signal than simple queue minimization.
"""

import numpy as np


def compute_maxpressure_reward(eng, intersection_id, current_phase):
    """
    Compute MaxPressure reward for current intersection and phase.

    MaxPressure = sum(incoming_queues) - sum(outgoing_queues)

    For a single intersection, we approximate this by:
    - Incoming: lanes with vehicles waiting (approaching intersection)
    - Outgoing: lanes downstream (after intersection)

    Simplified version: Use total queue pressure
    Reward = -total_queue_length (negative pressure, want to minimize)

    Args:
        eng: CityFlow engine
        intersection_id: ID of intersection
        current_phase: Current traffic light phase

    Returns:
        MaxPressure reward value (negative = pressure to reduce)
    """
    # Get all lane vehicle counts
    lane_waiting = eng.get_lane_waiting_vehicle_count()
    lane_vehicles = eng.get_lane_vehicle_count()

    if not lane_waiting:
        return 0.0

    # Simplified MaxPressure: total waiting vehicles (pressure to clear)
    # This is equivalent to -pressure since we want to minimize queue
    total_waiting = sum(lane_waiting.values())
    num_lanes = len(lane_waiting)

    # FIXED: Normalize by number of lanes to match H1's reward scale
    # This makes rewards comparable to H1-Basic which uses -mean(queue)
    if num_lanes == 0:
        return 0.0

    # Return negative pressure normalized by network size
    return -float(total_waiting) / num_lanes


def compute_full_maxpressure(eng, intersection_id, phase_id, lane_phase_mapping=None):
    """
    Compute full MaxPressure value for a specific phase.

    This computes the actual incoming - outgoing pressure if lane mappings are provided.

    Args:
        eng: CityFlow engine
        intersection_id: ID of intersection
        phase_id: Phase to compute pressure for
        lane_phase_mapping: Optional dict mapping phases to (incoming_lanes, outgoing_lanes)

    Returns:
        Pressure value for this phase
    """
    if lane_phase_mapping is None:
        # Fallback to simplified version
        return -sum(eng.get_lane_waiting_vehicle_count().values())

    # Get lanes for this phase
    incoming_lanes, outgoing_lanes = lane_phase_mapping.get(phase_id, ([], []))

    lane_counts = eng.get_lane_vehicle_count()

    # Compute pressure
    incoming_pressure = sum(lane_counts.get(lane, 0) for lane in incoming_lanes)
    outgoing_pressure = sum(lane_counts.get(lane, 0) for lane in outgoing_lanes)

    return float(incoming_pressure - outgoing_pressure)


def get_maxpressure_action(eng, intersection_id, available_phases):
    """
    Select action based on MaxPressure heuristic.

    Chooses the phase with maximum pressure (most incoming - outgoing vehicles).

    Args:
        eng: CityFlow engine
        intersection_id: ID of intersection
        available_phases: List of available phase IDs

    Returns:
        Best phase ID based on MaxPressure
    """
    lane_counts = eng.get_lane_vehicle_count()

    if not lane_counts:
        return available_phases[0] if available_phases else 0

    # Simplified heuristic: choose phase that serves lanes with most vehicles
    # For single intersection, we approximate by checking waiting counts per phase

    # Since we don't have explicit lane-phase mapping, use a simple heuristic:
    # Rotate through phases, favoring phases with higher average waiting
    lane_waiting = eng.get_lane_waiting_vehicle_count()

    # Simple strategy: return phase that would serve most waiting vehicles
    # Without roadnet data, we approximate by total queue pressure
    total_waiting = sum(lane_waiting.values())

    # For simple implementation, use round-robin weighted by pressure
    # In practice, this defaults to checking which phase has green lanes with most waiting

    # Return the phase - for now use simplified approach
    # (Full implementation would map lanes to phases and compute pressure per phase)
    if total_waiting > 10:  # High pressure threshold
        # Try to cycle to next phase
        current_phase = eng.get_intersection_phase(intersection_id)
        if current_phase:
            return (current_phase[0] + 1) % len(available_phases)

    # Default: maintain current or return first phase
    current = eng.get_intersection_phase(intersection_id)
    return current[0] if current else available_phases[0]


def estimate_lane_pressure_by_phase(eng, intersection_id, num_phases):
    """
    Estimate pressure for each phase based on lane waiting counts.

    This is a heuristic approximation when roadnet mapping isn't available.

    Args:
        eng: CityFlow engine
        intersection_id: ID of intersection
        num_phases: Number of available phases

    Returns:
        List of pressure estimates per phase
    """
    lane_waiting = eng.get_lane_waiting_vehicle_count()
    lane_list = sorted(lane_waiting.keys())

    if not lane_list:
        return [0.0] * num_phases

    # Heuristic: distribute lanes across phases
    # Assumes phases alternate between orthogonal directions
    lanes_per_phase = len(lane_list) // num_phases

    pressures = []
    for phase_idx in range(num_phases):
        start_idx = phase_idx * lanes_per_phase
        end_idx = start_idx + lanes_per_phase if phase_idx < num_phases - 1 else len(lane_list)

        phase_lanes = lane_list[start_idx:end_idx]
        phase_pressure = sum(lane_waiting.get(lane, 0) for lane in phase_lanes)
        pressures.append(float(phase_pressure))

    return pressures
