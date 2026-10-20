import sys
from pathlib import Path

# The ML code (features, predict, and train, which the saved models pickle as) lives in ml/src.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml" / "src"))
