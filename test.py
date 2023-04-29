import sys
import os
from pathlib import Path
print(os.path.dirname(__file__))
sys.path.insert(0, str(Path(__file__).resolve().parent / "lama"))