

import logging
from pathlib import Path
from mintransformer.main import main

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    main(Path("data/names.txt"))
