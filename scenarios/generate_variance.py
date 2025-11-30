"""
Generate traffic flow files with controlled variance for H1 testing.

Low-variance: Constant arrival rates with minimal variation
High-variance: Variable arrival rates with periodic surges
"""

import json
import numpy as np
from pathlib import Path

def generate_flow_with_variance(
    base_interval=5.0,
    variance_type='low',
    num_routes=12,
    output_file='flow.json',
    seed=42
):
    """
    Generate traffic flow file with controlled variance.

    Args:
        base_interval: Base vehicle spawn interval (seconds)
        variance_type: 'low' or 'high'
        num_routes: Number of routes (based on roadnet-adv.json)
        output_file: Output filename
        seed: Random seed for reproducibility
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

    # Define routes (matching roadnet-adv.json structure)
    routes = [
        ["road_0_1_0", "road_1_1_0"],  # West to East
        ["road_2_1_2", "road_1_1_2"],  # East to West
        ["road_1_0_1", "road_1_1_1"],  # South to North
        ["road_1_2_3", "road_1_1_3"],  # North to South
        ["road_1_0_1", "road_1_1_0"],  # South to East (turn)
        ["road_0_1_0", "road_1_1_1"],  # West to North (turn)
        ["road_2_1_2", "road_1_1_3"],  # East to South (turn)
        ["road_1_2_3", "road_1_1_2"],  # North to West (turn)
        ["road_0_1_0", "road_1_1_3"],  # West to South (turn)
        ["road_1_2_3", "road_1_1_0"],  # North to East (turn)
        ["road_2_1_2", "road_1_1_1"],  # East to North (turn)
        ["road_1_0_1", "road_1_1_2"],  # South to West (turn)
    ]

    flow = []

    if variance_type == 'low':
        # Low variance: Constant intervals with small noise (σ=0.5s)
        for route in routes:
            # Add small random jitter to break perfect synchronization
            interval = base_interval + np.random.uniform(-0.5, 0.5)
            flow.append({
                "vehicle": vehicle_template,
                "route": route,
                "interval": round(interval, 2),
                "startTime": 0,
                "endTime": -1
            })

    elif variance_type == 'high':
        # High variance: Mix of different intervals
        for i, route in enumerate(routes):
            # Assign different intervals to different routes
            if i % 3 == 0:
                interval = base_interval * 0.5  # High flow
            elif i % 3 == 1:
                interval = base_interval * 1.5  # Low flow
            else:
                interval = base_interval  # Normal flow

            flow.append({
                "vehicle": vehicle_template,
                "route": route,
                "interval": round(interval, 2),
                "startTime": 0,
                "endTime": -1
            })

    else:
        raise ValueError(f"Unknown variance_type: {variance_type}")

    # Write to file
    output_path = Path(__file__).parent / 'configs' / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(flow, f, indent=2)

    print(f"Generated {variance_type}-variance flow: {output_path}")
    print(f"  Routes: {len(flow)}")
    print(f"  Base interval: {base_interval}s")

    return str(output_path)

def generate_all_variance_scenarios():
    """Generate all variance scenarios for H1 testing."""
    scenarios = []

    # Low-variance scenario
    low_var = generate_flow_with_variance(
        base_interval=5.0,
        variance_type='low',
        output_file='flow_low_variance.json',
        seed=42
    )
    scenarios.append(('low_variance', low_var))

    # High-variance scenario
    high_var = generate_flow_with_variance(
        base_interval=5.0,
        variance_type='high',
        output_file='flow_high_variance.json',
        seed=42
    )
    scenarios.append(('high_variance', high_var))

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
    print("Generating variance-controlled traffic scenarios...\n")
    scenarios = generate_all_variance_scenarios()
    print(f"\n✅ Generated {len(scenarios)} variance scenarios")
