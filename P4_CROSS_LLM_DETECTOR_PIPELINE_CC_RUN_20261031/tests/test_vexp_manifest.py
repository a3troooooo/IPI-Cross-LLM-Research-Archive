import json
from pathlib import Path

def test_vexp_manifest():
    p=Path(__file__).resolve().parents[1]/'config'/'V_exp_manifest.json'
    d=json.loads(p.read_text())
    assert len(d['features'])==6
    assert len(set(d['features']))==6
