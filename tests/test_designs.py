"""The named designs D100 and D80 reproduce, bit for bit, the designs recorded in the original certificates."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)

from robbins_lb.designs import D80, D100, SPEC_D80, SPEC_D100, build  # noqa: E402

CERTS = os.path.join(ROOT, "certificates", "original")


@pytest.mark.parametrize("name,design,spec", [("n100_g20_down_f32", D100, SPEC_D100),
                                              ("n80_nocap_down_f32", D80, SPEC_D80)])
def test_named_design_matches_certificate(name, design, spec):
    cert = json.load(open(os.path.join(CERTS, name + ".cert.json")))
    assert cert["spec"] == spec
    assert design.to_dict() == cert["design"]          # floats compared exactly
    assert build(cert["spec"]).to_dict() == cert["design"]
    # exact state counts of the certified run (layers 0..n-2 are stored; layer n-1 is closed form)
    sizes = [design.layer_states(t) for t in range(design.n - 1)]
    assert max(sizes) == cert["max_layer_states"]
    assert sum(sizes) == cert["total_states"]
    assert sum(sizes) * (design.Gf + design.D) == cert["panels"]
