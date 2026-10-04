# wongutils

https://xkcd.com/1205/

Python utilities for simulation data, black hole calculations, and visualization.

## Install

```sh
pip install wongutils
```

## What's included

- `wongutils.geometry`: coordinate transforms, Kerr metrics, and interpolation.
- `wongutils.grmhd`: AthenaK and iHARM snapshots, meshblocks, and fluid utilities.
- `wongutils.grrt`: image filtering, synchrotron emission, and ipole helpers.
- `wongutils.photonring`: analytic photon orbits and critical curves.
- `wongutils.algorithms`: Newton's method and finite-difference Jacobians.
- `wongutils.visit`: tools for editing VisIt session and plot settings.

The ipole helpers require a separately installed ipole executable.

## Quickstart

```python
import numpy as np
from wongutils.geometry.metrics import get_gcov_ks_from_ks

gcov = get_gcov_ks_from_ks(0.5, 10.0, np.pi / 2)
print(gcov.shape)  # (4, 4)
```

## Development

```sh
pip install -e . pytest
pytest tests/
```

MIT licensed; see `LICENSE` in the repository.
