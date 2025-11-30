"""
Generate traffic flow files with periodic surges for H1 testing.

Surge scenarios: Base traffic with periodic high-density surges
"""

import json
import numpy as np
from pathlib import Path

def generate_surge_flow(
    base_interval=5.0,
    surge_interval=1.0,
    surge_duration=100,
    surge_period=300,
    num_surges=3,
    output_file='flow_surge.json',
    seed=42
):
    """
    Generate traffic flow with periodic surges.

    This creates multiple flow entries with different start/end times to simulate
    traffic surges at specific time periods.

    Args:
        base_interval: Normal traffic spawn interval (seconds)
        surge_interval: Surge traffic spawn interval (seconds)
        surge_duration: How long each surge lasts (simulation steps)
        surge_period: Time between surge starts (simulation steps)
        num_surges: Number of surge events
        output_file: Output filename
        seed: Random seed
    """
    np.random.seed(seed)

    # Vehicle template
    vehicle_template = {
        "length": 5.0,
        "width": 2.0,
        "maxPosAcc": 2.0,
        "maxNegAcc": 4.5,
        "usualPosAcc": 2.0,
        "usualNegAcc": 4.5,
        "minGap": 2.5,
        "maxSpeed": 16.67,
        "headwayTime": 1.5
    }

    # Define routes
    routes = [
        ["road_0_1_0", "road_1_1_0"],
        ["road_2_1_2", "road_1_1_2"],
        ["road_1_0_1", "road_1_1_1"],
        ["road_1_2_3", "road_1_1_3"],
        ["road_1_0_1", "road_1_1_0"],
        ["road_0_1_0", "road_1_1_1"],
        ["road_2_1_2", "road_1_1_3"],
        ["road_1_2_3", "road_1_1_2"],
        ["road_0_1_0", "road_1_1_3"],
        ["road_1_2_3", "road_1_1_0"],
        ["road_2_1_2", "road_1_1_1"],
        ["road_1_0_1", "road_1_1_2"],
    ]

    flow = []

    # Add base traffic (continuous)
    for route in routes:
        flow.append({
            "vehicle": vehicle_template,
            "route": route,
            "interval": base_interval,
            "startTime": 0,
            "endTime": -1  # Runs entire simulation
        })

    # Add surge traffic (periodic)
    # Each surge adds additional vehicles on top of base traffic
    for surge_idx in range(num_surges):
        surge_start = surge_idx * surge_period
        surge_end = surge_start + surge_duration

        # Only surge on some routes (to create localized congestion)
        surge_routes = routes[::3]  # Every 3rd route gets surge

        for route in surge_routes:
            flow.append({
                "vehicle": vehicle_template,
                "route": route,
                "interval": surge_interval,
                "startTime": surge_start,
                "endTime": surge_end
            })

    # Write to file
    output_path = Path(__file__).parent / 'configs' / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(flow, f, indent=2)

    print(f"Generated surge flow: {output_path}")
    print(f"  Base traffic: {len(routes)} routes @ {base_interval}s interval")
    print(f"  Surges: {num_surges} events, {len(surge_routes)} routes each")
    print(f"  Surge pattern: {surge_duration}s every {surge_period}s")

    return str(output_path)

def generate_all_surge_scenarios():
    """Generate all surge scenarios for H1 testing."""
    scenarios = []

    # Moderate surge scenario
    moderate = generate_surge_flow(
        base_interval=5.0,
        surge_interval=1.5,
        surge_duration=100,
        surge_period=300,
        num_surges=3,
        output_file='flow_moderate_surge.json',
        seed=42
    )
    scenarios.append(('moderate_surge', moderate))

    # Extreme surge scenario
    extreme = generate_surge_flow(
        base_interval=5.0,
        surge_interval=1.0,  # CityFlow requires interval >= 1.0
        surge_duration=150,
        surge_period=350,
        num_surges=3,
        output_file='flow_extreme_surge.json',
        seed=42
    )
    scenarios.append(('extreme_surge', extreme))

    # Create config files
    base_dir = Path(__file__).parent.parent
    for name, flow_file in scenarios:
        config = {
            "interval": 1.0,
            "seed": 42,
            "dir": str(base_dir / "data") + "/",
            "roadnetFile": "roadnet-adv.json",
            "flowFile": f"../scenarios/configs/flow_{name}.json",
            "rlTrafficLight": True,
            "laneChange": False,
            "saveReplay": False
        }

        config_path = Path(__file__).parent / 'configs' / f'config_{name}.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Created config: {config_path}")

    return scenarios

if __name__ == '__main__':
    print("Generating surge traffic scenarios...\n")
    scenarios = generate_all_surge_scenarios()
    print(f"\n✅ Generated {len(scenarios)} surge scenarios")
