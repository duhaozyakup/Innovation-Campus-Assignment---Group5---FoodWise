"""Create and execute the pure EDA notebook for FoodWise Assignment 1."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_eda_notebook import main as build_eda_main

if __name__ == "__main__":
    print("Executing scripts/build_notebook.py -> invoking EDA notebook builder...")
    build_eda_main()
