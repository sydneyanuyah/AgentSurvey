"""Run the full reproducible pipeline from the repository root."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def run(path: Path) -> None:
    print(f"\n>>> Running {path.relative_to(ROOT)}")
    subprocess.run([sys.executable, str(path)], cwd=ROOT, check=True)

if __name__ == "__main__":
    run(ROOT / "relevance/src/calculate_rrf_relevance.py")
    run(ROOT / "src/train_technical_depth.py")
    run(ROOT / "src/calculate_critic_ranking.py")
    print("\nPipeline completed successfully.")
