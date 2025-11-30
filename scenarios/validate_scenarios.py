"""
Validate that all generated scenarios work with CityFlow.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "CityFlow" / "build"))

import cityflow

def validate_scenario(config_path):
    """Test if a scenario config loads properly."""
    try:
        eng = cityflow.Engine(config_path, thread_num=1)

        # Run a few steps to ensure it works
        for _ in range(10):
            eng.next_step()

        vehicle_count = eng.get_vehicle_count()
        intersection_ids = eng.get_intersection_ids()

        print(f"✅ {config_path}")
        print(f"   Intersections: {len(intersection_ids)}")
        print(f"   Vehicles after 10 steps: {vehicle_count}")
        return True

    except Exception as e:
        print(f"❌ {config_path}")
        print(f"   Error: {e}")
        return False

def main():
    print("Validating traffic scenarios...\n")

    scenarios_dir = Path(__file__).parent / 'configs'
    config_files = sorted(scenarios_dir.glob('config_*.json'))

    results = {}
    for config_file in config_files:
        name = config_file.stem.replace('config_', '')
        results[name] = validate_scenario(str(config_file))
        print()

    # Summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    passed = sum(results.values())
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✅ All scenarios validated successfully!")
    else:
        print("\n⚠️  Some scenarios failed validation")
        for name, success in results.items():
            if not success:
                print(f"  - {name}")

    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
