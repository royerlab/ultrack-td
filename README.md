# ultrack-td

High-performance C++ implementation of [ultrack](https://github.com/royerlab/ultrack)'s multi-hypothesis segmentation algorithm, designed for the [tracksdata](https://github.com/royerlab/tracksdata) API.

## Overview

A C++ rewrite of ultrack's core segmentation algorithm providing an magnetude speedup while maintaining compatibility with both tracksdata and ultrack APIs. Implements multi-hypothesis segmentation with hierarchical watershed for robust cell segmentation in microscopy imaging.

## Installation

```bash
pip install ultrack-td
```

Or from source:

```bash
git clone https://github.com/royerlab/ultrack-td.git
cd ultrack-td
pip install .
```

**Requirements**: Python 3.10+, CMake 3.15+ (for building), C++17 compiler

## Quick Start

### tracksdata API (Recommended)

```python
import tracksdata as td
from ultrack_td import UltrackMultiHypotheses

# Initialize graph and segmenter
graph = td.graph.InMemoryGraph()
segmenter = UltrackMultiHypotheses(
    min_area=100,
    max_area=10000,
)

# Add segmentation nodes
segmenter.add_nodes(
    graph=graph,
    foreground=foreground,  # binary mask (T, Z, Y, X)
    detection=contours      # float boundary map
)

# Link and solve
td.edges.DistanceEdges(max_distance=50).add_edges(graph)
td.solvers.NearestNeighborsSolver().solve(graph)
```

### ultrack v1 API (Compatibility Mode)

```python
from ultrack_td.v1 import track
from ultrack.config import MainConfig

config = MainConfig()
config.segmentation_config.min_area = 100
config.segmentation_config.max_area = 10000

track(
    graph=graph,
    config=config,
    foreground=foreground,
    contours=contours
)
```

## Examples

Run the zebrafish tracking example:

```bash
python examples/zebrahub.py
```

## Limitations

**⚠️ This implementation supports only MINIMAL features from ultrack v1 API.**

### Not Implemented

The following ultrack features are **not supported**:

1. **Link Functions**: Only `power` link function supported (default)
   - Other link functions (`linear`, `exp`, etc.) not available

2. **Image Features**: The `images` parameter is not supported
   - Feature extraction from additional image channels not available
   - Use tracksdata API directly to add custom features

3. **Vector Fields**: Motion compensation via vector fields not supported
   - Apply coordinate shifts manually in tracksdata graph

4. **Window Size**: Temporal windowing for large datasets not supported
   - Use tracksdata's chunking API directly for large-scale processing

5. **Image Border Size**: Coordinate offsetting for image borders not supported
   - Apply border adjustments manually in tracksdata graph

6. **Overwrite Modes**: Only `overwrite="none"` or `overwrite=False` supported
   - Database management features not available

7. **Advanced Segmentation Options**: Some ultrack segmentation config options may not be exposed

### Recommended Migration Path

For full ultrack functionality, use the tracksdata API directly:

```python
import tracksdata as td
from ultrack_td import UltrackMultiHypotheses

# 1. Segmentation (this library)
segmenter = UltrackMultiHypotheses(...)
segmenter.add_nodes(graph, foreground, detection)

# 2. Add custom features manually
# your_custom_feature_extraction(graph)

# 3. Linking (tracksdata)
td.edges.DistanceEdges(...).add_edges(graph)

# 4. Custom edge attributes (tracksdata)
# your_custom_edge_attrs(graph)

# 5. Solving (tracksdata)
td.solvers.ILPSolver(...).solve(graph)
```

See [src/ultrack_td/v1.py](src/ultrack_td/v1.py) for implementation details and error messages.

## Citation

```bibtex
@inproceedings{bragantini2024ucmtracking,
  title={Large-scale multi-hypotheses cell tracking using ultrametric contours maps},
  author={Bragantini, Jord{\~a}o and Lange, Merlin and Royer, Lo{\"\i}c},
  booktitle={European Conference on Computer Vision},
  pages={36--54},
  year={2024},
  organization={Springer}
}

@article{bragantini2025ultrack,
  title={Ultrack: pushing the limits of cell tracking across biological scales},
  author={Bragantini, Jord{\~a}o and Theodoro, Ilan and Zhao, Xiang and Huijben, Teun APM and Hirata-Miyasaki, Eduardo and VijayKumar, Shruthi and Balasubramanian, Akilandeswari and Lao, Tiger and Agrawal, Richa and Xiao, Sheng and others},
  journal={Nature Methods},
  pages={1--14},
  year={2025},
  publisher={Nature Publishing Group US New York}
}
```

## Links

- [ultrack](https://github.com/royerlab/ultrack) - Original Python implementation
- [tracksdata](https://github.com/royerlab/tracksdata) - Graph-based tracking framework

## License

BSD-3-Clause
