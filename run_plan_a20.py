from pathlib import Path
import json
from src.plan_a20_data import build_plan_a20

if __name__ == "__main__":
    print(json.dumps(build_plan_a20(Path(__file__).resolve().parent),ensure_ascii=False,indent=2))
