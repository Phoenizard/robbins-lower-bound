import json
import os
import sys

import pytest

VERIFY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if VERIFY not in sys.path:
    sys.path.insert(0, VERIFY)

RESULTS = os.path.join(VERIFY, "results")

# Bruss & Ferguson (1993) Table 3, W_n^(m), as printed (4 decimals) -- index convention: BF's m
BF_TABLE3 = {
    1: {2: "1.2500", 3: "1.3242", 5: "1.3803", 9: "1.4171", 17: "1.4385"},
    2: {3: "1.3915", 5: "1.4991", 9: "1.5689", 17: "1.6092"},
    3: {5: "1.5509", 9: "1.6490", 17: "1.7070"},
    4: {5: "1.5710", 9: "1.6956", 17: "1.7668"},
    5: {9: "1.7260", 17: "1.8090"},
}
# BF Table 3, row "V_n": V_2, V_3, V_5
BF_VN = {2: "1.2500", 3: "1.3915", 5: "1.5710"}


def load_result(name):
    path = os.path.join(RESULTS, name + ".json")
    if not os.path.exists(path):
        pytest.skip(f"results/{name}.json not present (run: python run.py {name})")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
