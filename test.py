import sys
from pathlib import Path

# Ensure the locally built extension is discoverable when CityFlow isn't installed via pip.
sys.path.insert(0, str(Path(__file__).resolve().parent / "CityFlow" / "build"))

import cityflow

engine = cityflow.Engine("data/config.json", thread_num=1)
print("✅ CityFlow engine initialized!")
