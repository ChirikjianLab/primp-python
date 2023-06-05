# PRobabilistically-Informed Motion Primitives (PRIMP)
Python implementation of PRobabilistically-Informed Motion Primitives, a learning-from-demonstration method on Lie group.

Documentation is available in [ArXiv](https://arxiv.org/abs/2305.15761).

MATLAB version is available [here](https://github.com/ChirikjianLab/primp-matlab).

## Authors
Sipu Ruan (repository maintainer), Weixiao Liu, Xiaoli Wang, Xin Meng and Gregory S. Chirikjian

## Dependencies
See [requirements.txt](/primp/requirements.txt)
- numpy
- scipy
- [finitediff](https://pypi.org/project/finitediff/): Compute finite difference
- (Optional) [roboticstoolbox-python](https://github.com/petercorke/robotics-toolbox-python): Robotics toolbox for different operations and visualizations
- (Optional) [movement_primitives](https://github.com/dfki-ric/movement_primitives): Python library for movement primitives
- (Optional) [dtw-python](https://pypi.org/project/dtw-python/): Dynamic Time Warping for evaluating distance between learned and demonstrated trajectories

## Features
### PRIMP ([`/primp/src/primp/`](/primp/src/primp/))
Class for the proposed PRIMP method, working on Lie groups. The full 6D pose is considered.

### Probabilistic Movement Primitives (ProMP) ([`/primp/src/others/promp/`](`/primp/src/others/promp/`))
Wrapper class that calls the library [movement_primitives](https://github.com/dfki-ric/movement_primitives). The learning spaces include:

1. Only 3D position of the end effector
2. (TODO) the full 6D pose using [Orientation-ProMP](https://proceedings.mlr.press/v164/rozo22a.html)

## Usage
All scripts are located in [`/test`](/tests) folder.

### Using PRIMP for end effector 6D poses
```sh
python main_lfd_primp.py
```

### Using ProMP for end effector 3D positions
- Main script for ProMP encoding and conditioning
```sh
python main_lfd_promp.py
```

- Benchmark script for evaluations
```sh
python benchmark_lfd_promp.py
```

### Generated files
After running, 3 files will be generated (stored in `/result/${method}_${planning_group}/`):
1. `reference_density_${object}_${demo_type}.json`: Full information of the learned workspace trajectory distribution
2. `reference_density_${object}_${demo_type}_mean.csv`: Stores only the mean, for seeding the STOMP planner
3. `samples_${object}_${demo_type}.json`: Random samples from the learned trajectory distribution 