# Human-Robot Collision Avoidance — Digital Twin (UR16e)

> Proactive collision avoidance framework integrating real-time 3D human pose estimation,  
> motion prediction, and GPU-accelerated trajectory replanning for safe human-robot collaboration.

📄 [Full Paper (NYU MS Capstone)](paper/NYU25_Team6A_Report.pdf)

---

## Results

| Metric | Value |
|--------|-------|
| Collision avoidance rate | **100%** across 50 trials |
| Planning time (CPU) | 6–60 s |
| Planning time (GPU, CuPy) | **0.6–2.0 s** |
| Minimum human clearance | **≥275 mm** |
| Prediction-to-actuation latency | **<200 ms** |
| Control loop rate | 10 Hz |
| Test scenarios | Static, walking, partial occlusion |

---

## System Architecture
```
Orbbec RGB-D Camera + IMU
        ↓
3D Joint Deprojection (camera intrinsics + IMU tilt correction)
        ↓
MediaPipe Pose Estimation → 15 biomechanical landmarks
        ↓
Autoencoder (occlusion-robust joint recovery)
        ↓
BiLSTM Motion Forecaster (1s forecast from 3s window)
        ↓
APF Collision Risk Evaluation (capsule body model)
        ↓
A-RRT* Replanner (GPU-accelerated, bidirectional)
        ↓
ROS 2 Joint Trajectory Controller → UR16e
```

---

## My Contribution

- **A-RRT* motion planner** (`src/planning/Traj_plan_integration.py`) — full implementation including:
  - DH parameters and forward kinematics from scratch
  - Bidirectional tree search with Gaussian goal-biased sampling
  - APF-based node gating to discard unsafe configurations
  - Joint angle continuity via modulo 2π normalization
  - GPU acceleration via CuPy
- **3D perception pipeline** (`src/perception/visualization_motionprediction.py`) — including:
  - Live Orbbec RGB-D + IMU stream processing
  - IMU tilt correction (roll/pitch from accelerometer)
  - 3D joint deprojection using camera intrinsics
  - Occlusion handling via neighborhood depth averaging
  - ROS 2 MarkerArray publishing for RViz visualization

---

## Stack

`Python` `ROS 2` `CuPy` `PyTorch` `MediaPipe` `OpenCV` `Gazebo` `RViz` `Orbbec SDK`

---

## Repository Structure
```
├── src/
│   ├── planning/
│   │   └── Traj_plan_integration.py    # A-RRT* planner + ROS 2 integration
│   └── perception/
│       └── visualization_motionprediction.py  # Camera pipeline + pose estimation
├── paper/
│   └── NYU25_Team6A_Report.pdf
└── README.md
```

---

## Key Algorithms

### A-RRT* with APF Gating
Nodes are only added to the tree if their APF score is below a safety threshold τ = 20,
eliminating unsafe configurations during expansion and reducing computation in collision-prone regions.

### GPU Acceleration (CuPy)
Forward kinematics, APF evaluation, and capsule collision checks are vectorized and 
offloaded to GPU — reducing planning time from 6–60s (CPU) to 0.6–2.0s.

### IMU Tilt Correction
Accelerometer data from the Orbbec IMU computes roll (α) and pitch (β), 
applied as rotation matrices Rx(α) and Ry(β) to transform joint positions to world coordinates.
