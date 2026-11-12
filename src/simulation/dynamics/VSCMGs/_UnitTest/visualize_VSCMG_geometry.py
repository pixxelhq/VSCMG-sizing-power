import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

skew_angle = 55  # [deg]
azimuth_angles = [0, 90, 180, 270]  # [deg]

# VSCMG positions
vscmg_positions = [
    [0.4, 0.0, 0.0],   # VSCMG 0
    [0.0, 0.4, 0.0],   # VSCMG 1
    [-0.4, 0.0, 0.0],  # VSCMG 2
    [0.0, -0.4, 0.0]   # VSCMG 3
]

# Compute axes for all VSCMGs using loop
vscmgs = []
for i, az in enumerate(azimuth_angles):
    # Keep the exact same math pattern as original code
    if i % 2 == 0:  # VSCMG 0, 2
        gg = np.array([-np.cos(np.deg2rad(90-skew_angle))*np.cos(np.deg2rad(az)),
                       np.cos(np.deg2rad(90-skew_angle))*np.sin(np.deg2rad(az)),
                       np.sin(np.deg2rad(90-skew_angle))])
    else:  # VSCMG 1, 3
        gg = np.array([np.cos(np.deg2rad(90-skew_angle))*np.cos(np.deg2rad(az)),
                       -np.cos(np.deg2rad(90-skew_angle))*np.sin(np.deg2rad(az)),
                       np.sin(np.deg2rad(90-skew_angle))])
    
    # Same gs_hat for all
    gs = np.array([np.cos(np.deg2rad(skew_angle))*np.cos(np.deg2rad(az)),
                   np.cos(np.deg2rad(skew_angle))*np.sin(np.deg2rad(az)),
                   np.sin(np.deg2rad(skew_angle))])
    
    gt = np.cross(gg, gs)
    
    # Normalize
    gg_hat = gg / np.linalg.norm(gg)
    gs_hat = gs / np.linalg.norm(gs)
    gt_hat = gt / np.linalg.norm(gt)
    
    vscmgs.append({
        'index': i,
        'az': az,
        'pos': vscmg_positions[i],
        'gg': gg_hat,
        'gs': gs_hat,
        'gt': gt_hat
    })

# Create 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot all VSCMGs using loop
for vscmg in vscmgs:
    pos = vscmg['pos']
    
    # Plot gimbal axis (red)
    ax.quiver(pos[0], pos[1], pos[2], vscmg['gg'][0], vscmg['gg'][1], vscmg['gg'][2], 
             color='red', arrow_length_ratio=0.2, linewidth=2.5, alpha=0.8)
    
    # Plot spin axis (blue)
    ax.quiver(pos[0], pos[1], pos[2], vscmg['gs'][0], vscmg['gs'][1], vscmg['gs'][2], 
             color='blue', arrow_length_ratio=0.2, linewidth=2.5, alpha=0.8)
    
    # Plot transverse axis (green)
    ax.quiver(pos[0], pos[1], pos[2], vscmg['gt'][0], vscmg['gt'][1], vscmg['gt'][2], 
             color='green', arrow_length_ratio=0.2, linewidth=2.5, alpha=0.8)
    
    # Label each VSCMG
    label_pos = np.array(pos) + np.array([0.1, 0.1, 0.2])
    ax.text(label_pos[0], label_pos[1], label_pos[2], f'VSCMG {vscmg["index"]}\n({vscmg["az"]}°)', 
           fontsize=9, ha='center', weight='bold')

# Plot coordinate frame axes (black dashed)
ax.quiver(0, 0, 0, 0.6, 0, 0, color='black', arrow_length_ratio=0.15, linewidth=1.5, linestyle='--', alpha=0.6)
ax.quiver(0, 0, 0, 0, 0.6, 0, color='black', arrow_length_ratio=0.15, linewidth=1.5, linestyle='--', alpha=0.6)
ax.quiver(0, 0, 0, 0, 0, 0.6, color='black', arrow_length_ratio=0.15, linewidth=1.5, linestyle='--', alpha=0.6)

# Set labels and limits
ax.set_xlabel('X', fontsize=11, weight='bold')
ax.set_ylabel('Y', fontsize=11, weight='bold')
ax.set_zlabel('Z', fontsize=11, weight='bold')
ax.set_xlim([-0.8, 0.8])
ax.set_ylim([-0.8, 0.8])
ax.set_zlim([-0.2, 1.0])

ax.set_title(f'VSCMG Configuration - 4 VSCMGs (Skew = {skew_angle}°)', fontsize=12, weight='bold')

# Add legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='red', lw=2.5, label='Gimbal (gg)'),
                   Line2D([0], [0], color='blue', lw=2.5, label='Spin (gs)'),
                   Line2D([0], [0], color='green', lw=2.5, label='Transverse (gt)'),
                   Line2D([0], [0], color='black', lw=1.5, linestyle='--', label='Body frame')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# +++++++++++++++++++++++++++++++++++++++++
# Calculate angles for all VSCMGs

print(f"\nAngle Analysis for all VSCMGs (Skew = {skew_angle}°):\n")

# Define reference axes for each VSCMG
reference_axes = [
    np.array([1, 0, 0]),   # VSCMG 0: +X axis
    np.array([0, 1, 0]),   # VSCMG 1: +Y axis
    np.array([-1, 0, 0]),  # VSCMG 2: -X axis
    np.array([0, -1, 0])   # VSCMG 3: -Y axis
]

axis_labels = ['+X', '+Y', '-X', '-Y']

for vscmg, ref_axis, axis_label in zip(vscmgs, reference_axes, axis_labels):
    # Calculate angles using dot product
    angle_gg_ref = np.arccos(np.clip(np.dot(vscmg['gg'], ref_axis), -1, 1))
    angle_gs_ref = np.arccos(np.clip(np.dot(vscmg['gs'], ref_axis), -1, 1))
    angle_gs_gg = np.arccos(np.clip(np.dot(vscmg['gs'], vscmg['gg']), -1, 1))
    
    # Convert to degrees
    angle_gg_ref_deg = np.degrees(angle_gg_ref)
    angle_gs_ref_deg = np.degrees(angle_gs_ref)
    angle_gs_gg_deg = np.degrees(angle_gs_gg)
    
    print(f"VSCMG {vscmg['index']} (Azimuth = {vscmg['az']}°, Reference = {axis_label} axis):")
    print(f"  Angle gg vs {axis_label} axis: {angle_gg_ref_deg:.2f}°")
    print(f"  Angle gs vs {axis_label} axis: {angle_gs_ref_deg:.2f}°")
    print(f"  Angle gs vs gg: {angle_gs_gg_deg:.2f}° (should be ~90°)")
    print()


# import numpy as np
# import matplotlib.pyplot as plt

# skew_angle = 55 #degrees
# azimuth_angles = [0,90,180,270]

# # VSCMG positions
# vscmg_positions = [
#     [0.4, 0.0, 0.0],   # VSCMG 0
#     [0.0, 0.4, 0.0],   # VSCMG 1
#     [-0.4, 0.0, 0.0],  # VSCMG 2
#     [0.0, -0.4, 0.0]   # VSCMG 3
# ]

# # vscmg 1
# gg_1 = np.array([-np.cos(np.deg2rad(90-skew_angle))*np.cos(np.deg2rad(azimuth_angles[0])),
#                      np.cos(np.deg2rad(90-skew_angle))*np.sin(np.deg2rad(azimuth_angles[0])),
#                      np.sin(np.deg2rad(90-skew_angle)) ])
# gs_1 = np.array([np.cos(np.deg2rad(skew_angle))*np.cos(np.deg2rad(azimuth_angles[0])),
#                      np.cos(np.deg2rad(skew_angle))*np.sin(np.deg2rad(azimuth_angles[0])),
#                      np.sin(np.deg2rad(skew_angle))])
# gt_1 = np.cross(gg_1, gs_1)

# #vscmg 2
# gg_2 = np.array([np.cos(np.deg2rad(90-skew_angle))*np.cos(np.deg2rad(azimuth_angles[1])),
#                     -np.cos(np.deg2rad(90-skew_angle))*np.sin(np.deg2rad(azimuth_angles[1])),
#                     np.sin(np.deg2rad(90-skew_angle)) ])
# gs_2 = np.array([np.cos(np.deg2rad(skew_angle))*np.cos(np.deg2rad(azimuth_angles[1])),
#                      np.cos(np.deg2rad(skew_angle))*np.sin(np.deg2rad(azimuth_angles[1])),
#                      np.sin(np.deg2rad(skew_angle))])
# gt_2 = np.cross(gg_2, gs_2)

# # vscmg 3
# gg_3 = np.array([-np.cos(np.deg2rad(90-skew_angle))*np.cos(np.deg2rad(azimuth_angles[2])),
#                     np.cos(np.deg2rad(90-skew_angle))*np.sin(np.deg2rad(azimuth_angles[2])),
#                     np.sin(np.deg2rad(90-skew_angle))])
# gs_3 = np.array([np.cos(np.deg2rad(skew_angle))*np.cos(np.deg2rad(azimuth_angles[2])),
#                      np.cos(np.deg2rad(skew_angle))*np.sin(np.deg2rad(azimuth_angles[2])),
#                      np.sin(np.deg2rad(skew_angle))])
# gt_3 = np.cross(gg_3, gs_3)

# # vscmg 4
# gg_4 = np.array([np.cos(np.deg2rad(90-skew_angle))*np.cos(np.deg2rad(azimuth_angles[3])),
#                     -np.cos(np.deg2rad(90-skew_angle))*np.sin(np.deg2rad(azimuth_angles[3])),
#                     np.sin(np.deg2rad(90-skew_angle)) ])
# gs_4 = np.array([np.cos(np.deg2rad(skew_angle))*np.cos(np.deg2rad(azimuth_angles[3])),
#                      np.cos(np.deg2rad(skew_angle))*np.sin(np.deg2rad(azimuth_angles[3])),
#                      np.sin(np.deg2rad(skew_angle))])
# gt_4 = np.cross(gg_4, gs_4)


# # Compute axes for all VSCMGs using loop
# vscmgs = []
# for i, az in enumerate(azimuth_angles):
#     # Keep the exact same math pattern as your original code
#     if i % 2 == 0:  # VSCMG 0, 2
#         gg = np.array([-np.cos(np.deg2rad(90-skew_angle))*np.cos(np.deg2rad(az)),
#                        np.cos(np.deg2rad(90-skew_angle))*np.sin(np.deg2rad(az)),
#                        np.sin(np.deg2rad(90-skew_angle))])
#     else:  # VSCMG 1, 3
#         gg = np.array([np.cos(np.deg2rad(90-skew_angle))*np.cos(np.deg2rad(az)),
#                        -np.cos(np.deg2rad(90-skew_angle))*np.sin(np.deg2rad(az)),
#                        np.sin(np.deg2rad(90-skew_angle))])
    
#     # Same gs_hat for all
#     gs = np.array([np.cos(np.deg2rad(skew_angle))*np.cos(np.deg2rad(az)),
#                    np.cos(np.deg2rad(skew_angle))*np.sin(np.deg2rad(az)),
#                    np.sin(np.deg2rad(skew_angle))])
    
#     gt = np.cross(gg, gs)
    
#     # Normalize
#     gg_hat = gg / np.linalg.norm(gg)
#     gs_hat = gs / np.linalg.norm(gs)
#     gt_hat = gt / np.linalg.norm(gt)
    
#     vscmgs.append({
#         'index': i,
#         'az': az,
#         'pos': vscmg_positions[i],
#         'gg': gg_hat,
#         'gs': gs_hat,
#         'gt': gt_hat
#     })

# # normalize
# gg_hat_1 = gg_1/np.linalg.norm(gg_1)
# gs_hat_1 = gs_1/np.linalg.norm(gs_1)
# gt_hat_1 = gt_1/np.linalg.norm(gt_1)

# gg_hat_2 = gg_2/np.linalg.norm(gg_2)
# gs_hat_2 = gs_2/np.linalg.norm(gs_2)
# gt_hat_2 = gt_2/np.linalg.norm(gt_2)

# gg_hat_3 = gg_3/np.linalg.norm(gg_3)
# gs_hat_3 = gs_3/np.linalg.norm(gs_3)
# gt_hat_3 = gt_3/np.linalg.norm(gt_3)

# gg_hat_4 = gg_4/np.linalg.norm(gg_4)
# gs_hat_4 = gs_4/np.linalg.norm(gs_4)
# gt_hat_4 = gt_4/np.linalg.norm(gt_4)


# # Create 3D plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot gimbal axis (red)
# ax.quiver(vscmg_positions[0][0], vscmg_positions[0][1], vscmg_positions[0][2], gg_hat_1[0], gg_hat_1[1], gg_hat_1[2], 
#          color='red', arrow_length_ratio=0.2, linewidth=2.5, label='Gimbal (gg)')

# # Plot spin axis (blue)
# ax.quiver(vscmg_positions[0][0], vscmg_positions[0][1], vscmg_positions[0][2], gs_hat_1[0], gs_hat_1[1], gs_hat_1[2], 
#          color='blue', arrow_length_ratio=0.2, linewidth=2.5, label='Spin (gs)')

# # Plot transverse axis (green)
# ax.quiver(vscmg_positions[0][0], vscmg_positions[0][1], vscmg_positions[0][2], gt_hat_1[0], gt_hat_1[1], gt_hat_1[2], 
#          color='green', arrow_length_ratio=0.2, linewidth=2.5, label='Transverse (gt)')

# # Plot gimbal axis (red)
# ax.quiver(vscmg_positions[1][0], vscmg_positions[1][1], vscmg_positions[1][2], gg_hat_2[0], gg_hat_2[1], gg_hat_2[2], 
#          color='red', arrow_length_ratio=0.2, linewidth=2.5, label='Gimbal (gg)')

# # Plot spin axis (blue)
# ax.quiver(vscmg_positions[1][0], vscmg_positions[1][1], vscmg_positions[1][2], gs_hat_2[0], gs_hat_2[1], gs_hat_2[2], 
#          color='blue', arrow_length_ratio=0.2, linewidth=2.5, label='Spin (gs)')

# # Plot transverse axis (green)
# ax.quiver(vscmg_positions[1][0], vscmg_positions[1][1], vscmg_positions[1][2], gt_hat_2[0], gt_hat_2[1], gt_hat_2[2], 
#          color='green', arrow_length_ratio=0.2, linewidth=2.5, label='Transverse (gt)')

# # Plot gimbal axis (red)
# ax.quiver(vscmg_positions[2][0], vscmg_positions[2][1], vscmg_positions[2][2], gg_hat_3[0], gg_hat_3[1], gg_hat_3[2], 
#          color='red', arrow_length_ratio=0.2, linewidth=2.5, label='Gimbal (gg)')

# # Plot spin axis (blue)
# ax.quiver(vscmg_positions[2][0], vscmg_positions[2][1], vscmg_positions[2][2], gs_hat_3[0], gs_hat_3[1], gs_hat_3[2], 
#          color='blue', arrow_length_ratio=0.2, linewidth=2.5, label='Spin (gs)')

# # Plot transverse axis (green)
# ax.quiver(vscmg_positions[2][0], vscmg_positions[2][1], vscmg_positions[2][2], gt_hat_3[0], gt_hat_3[1], gt_hat_3[2], 
#          color='green', arrow_length_ratio=0.2, linewidth=2.5, label='Transverse (gt)')

# # Plot gimbal axis (red)
# ax.quiver(vscmg_positions[3][0], vscmg_positions[3][1], vscmg_positions[3][2], gg_hat_4[0], gg_hat_4[1], gg_hat_4[2], 
#          color='red', arrow_length_ratio=0.2, linewidth=2.5, label='Gimbal (gg)')

# # Plot spin axis (blue)
# ax.quiver(vscmg_positions[3][0], vscmg_positions[3][1], vscmg_positions[3][2], gs_hat_4[0], gs_hat_4[1], gs_hat_4[2], 
#          color='blue', arrow_length_ratio=0.2, linewidth=2.5, label='Spin (gs)')

# # Plot transverse axis (green)
# ax.quiver(vscmg_positions[3][0], vscmg_positions[3][1], vscmg_positions[3][2], gt_hat_4[0], gt_hat_4[1], gt_hat_4[2], 
#          color='green', arrow_length_ratio=0.2, linewidth=2.5, label='Transverse (gt)')

# # Plot coordinate frame axes (black dashed)
# ax.quiver(0, 0, 0, 1, 0, 0, color='black', arrow_length_ratio=0.15, linewidth=1.5, linestyle='--', alpha=0.6)
# ax.quiver(0, 0, 0, 0, 1, 0, color='black', arrow_length_ratio=0.15, linewidth=1.5, linestyle='--', alpha=0.6)
# ax.quiver(0, 0, 0, 0, 0, 1, color='black', arrow_length_ratio=0.15, linewidth=1.5, linestyle='--', alpha=0.6)

# # Set labels and limits
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.set_zlim([-1, 1])

# ax.set_title(f'VSCMG 0 Axes (Skew = {skew_angle}°)')
# ax.legend()
# ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()


# # +++++++++++++++++++++++++++++++++++++++++

# # Positive X-axis
# x_axis = np.array([1, 0, 0])

# # Calculate angles using dot product
# # cos(θ) = (u · v) / (|u| * |v|)
# # Since both are unit vectors, |u| = |v| = 1

# angle_gg_x = np.arccos(np.dot(gg_hat_1, x_axis))
# angle_gs_x = np.arccos(np.dot(gs_hat_1, x_axis))
# angle_gs_gg = np.arccos(np.dot(gs_hat_1, gg_hat_1))

# # Convert to degrees
# angle_gg_x_deg = np.degrees(angle_gg_x)
# angle_gs_x_deg = np.degrees(angle_gs_x)
# angle_gs_gg_deg = np.degrees(angle_gs_gg)

# print(f"Angle between gg_hat_1 and +X axis: {angle_gg_x_deg:.2f}°")
# print(f"Angle between gs_hat_1 and +X axis: {angle_gs_x_deg:.2f}°")
# print(f"Angle between gs_hat_1 and gg_hat_1: {angle_gs_gg_deg:.2f}°")
