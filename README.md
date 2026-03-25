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
