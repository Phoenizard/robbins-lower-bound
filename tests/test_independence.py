"""benchmark/, replication/ and supplementary/ are independent checks: none of their files imports robbins_lb."""
import ast
import glob
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def _imported_modules(path):
    tree = ast.parse(open(path, encoding="utf8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_independent_code_does_not_import_robbins_lb():
    files = []
    for sub in ("benchmark", "replication", "supplementary"):
        files += glob.glob(os.path.join(ROOT, sub, "**", "*.py"), recursive=True)
    assert len(files) > 20
    bad = [(f, m) for f in files for m in _imported_modules(f) if m.split(".")[0] == "robbins_lb"]
    assert bad == []
