# Velocity field estimation using floating sensor locations
![model_schematics](https://github.com/oura-tomoya/floating-sensors/blob/images/model_schematics.png)
Sample codes for machine-learning (ML) based velocity field estimation using floating sensor locations.
In the present mothod, the ML model is supposed to generate velocity fields so that the time variation of floating sensor motion is consistent with the given data of sensor locations.
For more details, please refer to the reference below.

## Reference
- T. Oura, R. Miura and K. Fukagata, "Machine-learning based flow field estimation using floating sensor locations," preprint, [arXiv:2311.08754](https://doi.org/10.48550/arXiv.2311.08754) (2025).

## Information
The estimation for the two-dimensional forced homogeneous isotropic turbulence (HIT) at $Re_{\lambda} \simeq 272$ is performed in the sample codes, which include both model training and model evaluation.
`data` directory contains
- `sensor_locations_at_n.npy`
    - Training dataset. Locations of 512 sensors for 2500 time-steps.
- `sensor_locations_at_n_plus_1.npy`
    - Training dataset. Sensor locations after one time-step from `sensor_locations_at_n.npy`.
- `ufvf_truth_at_*.npy`
    - Ground-truth data of two-dimensional velocity fields.
    - Due to storage limitations, only data every 200 time-steps is avaiable here.
    - Note that the ground-truth data is not used during model training. This is solely prepared for the model evaluation.

Authors provide no guarantees for this code. Use as-is and for academic research use only; no commercial use allowed without permission. The code is written for educational clarity and not for speed.

## Requirements
- Python 3.10.13
- TensorFlow 2.11.0
- NumPy 1.26.4
- matplotlib 3.8.4
