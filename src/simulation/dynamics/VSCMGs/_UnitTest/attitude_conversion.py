# Approach 1
from Basilisk.utilities import RigidBodyKinematics as rbk
import numpy as np

# Euler angles
euler_angles_321 = np.deg2rad([30, 5, 15])  # Example Euler angles in radians
print("Euler Angles (321 sequence):", np.rad2deg(euler_angles_321))

mrp = rbk.euler3212MRP(euler_angles_321)  # Convert Euler angles to MRP
print("MRP:", mrp)

euler_angles_321 = np.round(np.rad2deg(rbk.MRP2Euler321(mrp)), 3)  # Returns [angle1, angle2, angle3] in radians
print("Euler Angles (321 sequence):", euler_angles_321)


# # For the integrated code
# from Basilisk.utilities import RigidBodyKinematics as rbk

# # Convert all MRP data to Euler angles (321 sequence)
# eulerAngles = np.zeros_like(sigmaData)
# for i in range(len(sigmaData)):
#     eulerAngles[i, :] = rbk.MRP2Euler321(sigmaData[i, :])

# # Now plot Euler angles instead of MRP
# plt.figure()
# plt.plot(dataLog.times()*1e-9, eulerAngles * 180/np.pi)  # Convert to degrees
# plt.xlabel('Time (s)')
# plt.ylabel('Euler Angles (deg)')