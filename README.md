# MLPF
Robotic Fingertip Tactile Sensing for Human-Like Dexterous Manipulation: High-Fidelity Reconstruction of 3D Forces and Contact Localization

## Requirements
- Python 3.12
- PyTorch 2.6

## Code File Introduction
- Real_Time_Process.py:
  Real-time 3D Force and Force Application Point Coordinate Estimation
- Train_12axis_4regions.py: Training a Deep Learning Model to Achieve Four-Region Classification
- Train_mlp_12axis_4regions.py: Training a Deep Learning Model to Achieve Single-Region Fitting
- Train_only_mlp_12axis.py: Training a Deep Learning Model to Achieve Full-Region Fitting
- Trajectory_Recognition.py: Real-time Recognition of Sensor Sliding Trajectories on the Desktop
- Multi-sensor_Data_Acquisition.py: Multi-threaded Construction of Multi-sensor Communication
- Made_DataSet.py: A program enabling rapid dataset collection has been developed.
- Get_R.py: The rotation matrix is calibrated rapidly based on the proposed method.
- Demo_CNN.py: CNN Recognition of Letters Corresponding to Captured Trajectories
- Demo_KF_F.py: Data fusion of 3D force is realized via Kalman filter algorithm.
- Demo_KF_X.py: Kalman Filter Data Fusion for Force Application Point Coordinates
  
## Citation
If you use this code, please cite our paper.
