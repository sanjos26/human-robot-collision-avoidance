import numpy as np
poses = np.load('/home/nyu6a/Downloads/MSproj/live_predictions.npy')
poses_curr = np.load('/home/nyu6a/Downloads/MSproj/live_capture.npy')

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load predicted pose sequence

print(f"Loaded predictions with shape: {poses.shape}")

# Set up 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Setting limits based on your data (adjust if needed)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(0, 2)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.view_init(elev=30, azim=60)

# Scatter plot object
scat = ax.scatter([], [], [], c='red', s=30)

def update(frame):
    joints = poses[frame] / 1000.0  # assuming mm ➝ m
    joints = np.append(joints,poses_curr[frame]/1000.0,axis=0)
    scat._offsets3d = (joints[:, 0], joints[:, 1], joints[:, 2])
    colors = plt.cm.coolwarm(np.array([1]*15+[0]*15))
    scat.set_color(colors)
    return scat,

ani = animation.FuncAnimation(
    fig, update, frames=len(poses),
    interval=1000/30, blit=False
)

# 🎥 Save to file
ani.save("predicted_pose_video.mp4", fps=30, extra_args=['-vcodec', 'libx264'])

print("✅ Video saved as predicted_pose_video.mp4")
