from pathlib import Path
import json

directory = Path(__file__).parent

def get_secrets():
    with open(directory / "secrets.json", "r") as f:
        secrets = json.load(f)
    return secrets