

import logging
from pathlib import Path
from mintransformer.main import run_experiment

logging.basicConfig(level=logging.INFO)

run_experiment(Path("data/names.txt"))
