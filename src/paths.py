"""Central location for filesystem paths used across scripts and notebooks.

All datasets, memmaps, model checkpoints and experiment outputs live under
``DATA_ROOT``. By default this is ``<repo>/data``; override it by setting the
``FL_DENTAL_DATA_ROOT`` environment variable, e.g.

    export FL_DENTAL_DATA_ROOT=/path/to/your/data
"""

import os
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_ROOT: Path = Path(os.environ.get("FL_DENTAL_DATA_ROOT", REPO_ROOT / "data")).expanduser()
