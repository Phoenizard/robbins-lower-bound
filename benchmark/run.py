"""Entry point: python -I -X utf8 run.py <task> ...   (see robbins_bench/run_bench.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robbins_bench.run_bench import main  # noqa: E402

if __name__ == "__main__":
    main(sys.argv[1:])
