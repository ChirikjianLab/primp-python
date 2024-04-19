# PRobabilistically-Informed Motion Primitives (PRIMP)
[![Python](https://github.com/ChirikjianLab/primp-python/actions/workflows/python-package.yml/badge.svg)](https://github.com/ChirikjianLab/primp-python/actions/workflows/python-package.yml)

Python implementation of PRobabilistically-Informed Motion Primitives, a learning-from-demonstration method on Lie group. This work is published in _IEEE Transactions on Robotics (T-RO)_.

- Publication: [T-RO](https://ieeexplore.ieee.org/document/10502164)
- Project page: [https://chirikjianlab.github.io/primp-page/](https://chirikjianlab.github.io/primp-page/)
- MATLAB version (includes more demos): [https://github.com/ChirikjianLab/primp-matlab](https://github.com/ChirikjianLab/primp-matlab).

## Authors
[Sipu Ruan](https://ruansp.github.io), Weixiao Liu, Xiaoli Wang, Xin Meng and Gregory S. Chirikjian

## Dependencies
See [requirements.txt](/primp/requirements.txt)
- numpy
- scipy
- [finitediff](https://pypi.org/project/finitediff/): Compute finite difference
- [movement_primitives](https://github.com/dfki-ric/movement_primitives): Python library for movement primitives
- [dtw-python](https://pypi.org/project/dtw-python/): Dynamic Time Warping for evaluating distance between learned and demonstrated trajectories
- [roboticstoolbox-python](https://github.com/petercorke/robotics-toolbox-python): Robotics toolbox for different operations and visualizations

## Features
### PRIMP
Class for the proposed PRIMP method, working on Lie groups. The full 6D pose is considered.

### Probabilistic Movement Primitives (ProMP)
Wrapper class that calls the library [movement_primitives](https://github.com/dfki-ric/movement_primitives). The learning spaces include:

- Only 3D position of the end effector

## Installation
We recommend using `pip` to install the package:
```
pip install .
```

## Usage
### Data preparation for LfD methods
All test files are located in [/test](/test) folder. To run scripts for LfD methods:

- Download the data from [Google Drive](https://drive.google.com/drive/folders/1sgfAjBgO3PWO2nCqerXjVHsovpNF4MgS?usp=sharing). All the demonstrated datasets are locataed in `/demonstrations` folder.
- Generate `/data` folder that stores all demonstration data
- Copy all the demonstration sets into the `/data` folder
- Run scripts in /test folder


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

## Citation
```
S. Ruan, W. Liu, X. Wang, X. Meng and G. S. Chirikjian, "PRIMP: PRobabilistically-Informed Motion Primitives for Efficient Affordance Learning from Demonstration," in IEEE Transactions on Robotics, doi: 10.1109/TRO.2024.3390052.
```

BibTex
```
@ARTICLE{10502164,
  author={Ruan, Sipu and Liu, Weixiao and Wang, Xiaoli and Meng, Xin and Chirikjian, Gregory S.},
  journal={IEEE Transactions on Robotics}, 
  title={PRIMP: PRobabilistically-Informed Motion Primitives for Efficient Affordance Learning from Demonstration}, 
  year={2024},
  volume={},
  number={},
  pages={1-20},
  keywords={Trajectory;Robots;Probabilistic logic;Planning;Affordances;Task analysis;Manifolds;Learning from Demonstration;Probability and Statistical Methods;Motion and Path Planning;Service Robots},
  doi={10.1109/TRO.2024.3390052}}
```
