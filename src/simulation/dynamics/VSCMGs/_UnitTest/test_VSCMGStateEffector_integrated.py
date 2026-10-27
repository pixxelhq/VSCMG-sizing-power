
# ISC License
#
# Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.


import inspect
import math
import os

import numpy as np
import pandas as pd
import pytest

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))

from Basilisk.utilities import simHelpers
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import unitTestSupport  # general support file with common unit test functions
import matplotlib as mpl
import matplotlib.pyplot as plt
from Basilisk.simulation import spacecraft
from Basilisk.utilities import macros
from Basilisk.simulation import gravityEffector
from Basilisk.simulation import vscmgStateEffector
from Basilisk.architecture import messaging
from Basilisk.architecture.sysModel import SysModel

mpl.rc("figure", figsize=(5.75,4))


def defaultVSCMG():
    VSCMG = messaging.VSCMGConfigMsgPayload()

    VSCMG.rGB_B = [[0.],[0.],[0.]]
    VSCMG.gsHat0_B = [[0.],[0.],[0.]]
    VSCMG.gtHat0_B = [[0.],[0.],[0.]]
    VSCMG.ggHat_B = [[0.],[0.],[0.]]
    VSCMG.u_s_max = -1
    VSCMG.u_s_min = -1
    VSCMG.u_s_f = 0.
    VSCMG.wheelLinearFrictionRatio = -1
    VSCMG.u_g_current = 0.
    VSCMG.u_g_max = -1
    VSCMG.u_g_min = -1
    VSCMG.u_g_f = 0.
    VSCMG.gimbalLinearFrictionRatio = -1
    VSCMG.Omega = 0.
    VSCMG.gamma = 0.
    VSCMG.gammaDot = 0.
    VSCMG.Omega_max = 6000. * macros.RPM
    VSCMG.gammaDot_max = -1
    VSCMG.IW1 = 100./VSCMG.Omega_max # 0.159154943092
    VSCMG.IW2 = 0.5*VSCMG.IW1 # 0.0795774715459
    VSCMG.IW3 = 0.5*VSCMG.IW1 # 0.0795774715459
    VSCMG.IG1 = 0.1
    VSCMG.IG2 = 0.2
    VSCMG.IG3 = 0.3
    VSCMG.U_s = 4.8e-06 * 1e4
    VSCMG.U_d = 1.54e-06 * 1e4
    VSCMG.l = 0.01
    VSCMG.L = 0.1
    VSCMG.rGcG_G = [[0.0001],[-0.02],[0.1]]
    VSCMG.massW = 6.
    VSCMG.massG = 6.
    VSCMG.VSCMGModel = 0
    return VSCMG

def computeFFT(y,dt):
    Fs = 1.0/dt  # sampling rate
    Ts = dt # sampling interval
    n = len(y) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[list(range(n//2))] # one side frequency range
    Y = np.fft.fft(y)/n # fft computing and normalization
    Y = Y[list(range(n//2))]
    Y = abs(Y)
    return [frq,Y]

def findPeaks(Y,maxfind):
        peakIdxs = np.r_[True, Y[1:] > Y[:-1]] & np.r_[Y[:-1] > Y[1:], True]
        peakIdxs[0] = False
        peakIdxs[-1] = False

        peakIdxs = np.array(np.where(peakIdxs==True))[0]
        threshold = np.sort(Y[peakIdxs])[-maxfind]
        peakIdxs = peakIdxs[np.where(Y[peakIdxs] >= threshold)[0]]

        return peakIdxs

def mrp2Quaternion(sigma):
    """Convert Modified Rodrigues Parameter to Quaternion [q0, q1, q2, q3]"""
    sigma_norm = np.linalg.norm(sigma)
    if sigma_norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    q0 = (1 - sigma_norm**2) / (1 + sigma_norm**2)
    q_vec = 2 * sigma / (1 + sigma_norm**2)
    return np.array([q0, q_vec[0], q_vec[1], q_vec[2]])

def mrp2EulerAngles(sigma):
    """Convert Modified Rodrigues Parameter to Euler angles [roll, pitch, yaw] in radians"""
    sigma_norm = np.linalg.norm(sigma)
    if sigma_norm < 1e-10:
        return np.array([0.0, 0.0, 0.0])

    q0 = (1 - sigma_norm**2) / (1 + sigma_norm**2)
    q_vec = 2 * sigma / (1 + sigma_norm**2)
    q1, q2, q3 = q_vec[0], q_vec[1], q_vec[2]

    # Roll (phi)
    roll = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))

    # Pitch (theta)
    pitch = np.arcsin(2*(q0*q2 - q3*q1))

    # Yaw (psi)
    yaw = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))

    return np.array([roll, pitch, yaw])


@pytest.mark.parametrize("useFlag, testCase", [
    (False,'BalancedWheels'),
    (False,'JitterSimple'),
    (False,'JitterFullyCoupled'),
    (False,'JitterFullyCoupledGravity'),
])

# uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed
# @pytest.mark.skipif(conditionstring)
# uncomment this line if this test has an expected failure, adjust message as needed
# @pytest.mark.xfail() # need to update how the RW states are defined
# provide a unique test method name, starting with test_
def test_VSCMGIntegratedTest(show_plots,useFlag,testCase):
    """Module Unit Test"""
    [testResults, testMessage] = VSCMGIntegratedTest(show_plots,useFlag,testCase)
    assert testResults < 1, testMessage

# Reading the CSV
df = pd.read_csv(r"C:\Users\gavin.furtado\Desktop\Workspace Gavin\Projects\Ladybird\LBEph - LBEph.csv", nrows=0)  # nrows=0 reads only the header
print(list(df.columns))

def quaternion_to_mrp_optimal(q):
    """
    Convert quaternion to MRP with shadow set handling
    Returns the MRP with smaller magnitude (< 1)
    
    Args:
        q (array-like): Quaternion [q0, q1, q2, q3]
    
    Returns:
        sigma (ndarray): MRP [σ1, σ2, σ3]
    """
    import numpy as np
    
    q = np.atleast_1d(q)
    
    if q.ndim == 1:
        q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
        
        if abs(1 + q0) < 1e-10:
            return np.array([0.0, 0.0, 0.0])
        
        # Primary MRP
        sigma = np.array([q1, q2, q3]) / (1 + q0)
        
        # Check magnitude and use shadow set if needed
        sigma_norm = np.linalg.norm(sigma)
        if sigma_norm > 1.0:
            # Use shadow set: σ_shadow = -σ / ||σ||²
            sigma = -sigma / (sigma_norm ** 2)
        
        return sigma
    
    else:
        q0 = q[:, 0]
        q_vec = q[:, 1:4]
        
        denominator = 1 + q0
        denominator[abs(denominator) < 1e-10] = 1e-10
        
        sigma = q_vec / denominator[:, np.newaxis]
        
        # Handle shadow set for each MRP
        sigma_norms = np.linalg.norm(sigma, axis=1)
        shadow_mask = sigma_norms > 1.0
        sigma[shadow_mask] = -sigma[shadow_mask] / (sigma_norms[shadow_mask][:, np.newaxis] ** 2)
        
        return sigma

def roll_to_mrp(roll_rad):
    """
    Convert roll angle (rotation about X-axis) to MRP
    
    For pure roll rotation:
    θ = roll angle
    e = [1, 0, 0] (rotation axis)
    σ = tan(θ/4) * e = [tan(θ/4), 0, 0]
    """
    sigma = np.tan(roll_rad / 4.0)
    return np.array([sigma, 0.0, 0.0])


# Convert desired attitudes
roll_initial_deg = 0.0
roll_final_deg = 30.0

roll_initial_rad = np.radians(roll_initial_deg)
roll_final_rad = np.radians(roll_final_deg)

mrp_initial = roll_to_mrp(roll_initial_rad)
mrp_final = roll_to_mrp(roll_final_rad)

print(f"Initial roll: {roll_initial_deg}deg -> MRP: {mrp_initial}")
print(f"Final roll:   {roll_final_deg}deg -> MRP: {mrp_final}")

def generateRollManeuverProfile(roll_initial_deg, roll_final_deg, maneuver_time_sec, dt=0.1):
    """
    Generate smooth roll maneuver profile
    
    Args:
        roll_initial_deg: Initial roll angle [degrees]
        roll_final_deg: Final roll angle [degrees]
        maneuver_time_sec: Time to complete maneuver [seconds]
        dt: Time step [seconds]
    
    Returns:
        DataFrame with guidance profile
    """   
    # Convert to radians
    roll_i = np.radians(roll_initial_deg)
    roll_f = np.radians(roll_final_deg)
    
    # Time array
    time = np.arange(0, maneuver_time_sec + dt, dt)
    
    # Trapezoidal velocity profile
    # Accelerate for 1/3 time, coast for 1/3 time, decelerate for 1/3 time
    accel_time = maneuver_time_sec / 3.0
    coast_end = 2 * maneuver_time_sec / 3.0
    
    sigma_x = []
    omega_x = []
    
    for t in time:
        if t <= accel_time:
            # Accelerating phase
            omega = (roll_f - roll_i) / maneuver_time_sec * 3.0  # 0.088 rad/s for 30° over 2s
            roll = roll_i + 0.5 * omega * t**2 / accel_time
            omega_val = omega * t / accel_time
            
        elif t <= coast_end:
            # Coasting phase
            omega = (roll_f - roll_i) / maneuver_time_sec * 3.0
            roll = roll_i + (roll_f - roll_i) * (1/3 + (t - accel_time) / maneuver_time_sec)
            omega_val = omega
            
        else:
            # Decelerating phase
            t_decel = t - coast_end
            omega = (roll_f - roll_i) / maneuver_time_sec * 3.0
            roll = roll_f - 0.5 * omega * (maneuver_time_sec - t_decel)**2 / accel_time
            omega_val = omega * (1 - t_decel / accel_time)
        
        # Convert roll to MRP
        sigma_val = np.tan(roll / 4.0)
        sigma_x.append(sigma_val)
        omega_x.append(omega_val)
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': time,
        'sigma_x': sigma_x,
        'sigma_y': np.zeros_like(time),
        'sigma_z': np.zeros_like(time),
        'omega_x': omega_x,
        'omega_y': np.zeros_like(time),
        'omega_z': np.zeros_like(time)
    })
    
    return df

def calculateAngularAcceleration(guidance_df):
    """
    Calculate angular acceleration from angular velocity
    α = dω/dt
    """
    import numpy as np
    
    # Time and angular velocity arrays
    time = guidance_df['time'].values
    omega_x = guidance_df['omega_x'].values
    omega_y = guidance_df['omega_y'].values
    omega_z = guidance_df['omega_z'].values
    
    # Calculate time derivatives (finite difference)
    alpha_x = np.gradient(omega_x, time)  # dω_x/dt
    alpha_y = np.gradient(omega_y, time)  # dω_y/dt
    alpha_z = np.gradient(omega_z, time)  # dω_z/dt
    
    return alpha_x, alpha_y, alpha_z

def calculateBodyTorque(guidance_df, inertia_matrix):
    """
    Calculate body torque from angular velocity and inertia
    
    τ = I · α
    
    Args:
        guidance_df: DataFrame with time and omega values
        inertia_matrix: 3x3 spacecraft inertia matrix [kg·m²]
    
    Returns:
        DataFrame with torque components
    """
    import numpy as np
    
    # Calculate angular acceleration
    time = guidance_df['time'].values
    omega = guidance_df[['omega_x', 'omega_y', 'omega_z']].values
    
    # Time derivatives
    alpha = np.gradient(omega, time, axis=0)
    
    # Calculate torque: τ = I · α
    torque = np.zeros_like(alpha)
    for i in range(len(alpha)):
        torque[i] = inertia_matrix @ alpha[i]
    
    # Create output DataFrame
    result_df = guidance_df.copy()
    result_df['tau_x'] = torque[:, 0]
    result_df['tau_y'] = torque[:, 1]
    result_df['tau_z'] = torque[:, 2]
    
    return result_df



#==========================================
def buildTorqueAuthorityMatrix(VSCMGs):
    """
    Build torque authority matrix A where:
    τ_body = A @ u_gimbal
    
    Each column represents the torque contribution of one VSCMG gimbal command.
    
    Args:
        VSCMGs: List of VSCMG configurations
    
    Returns:
        A: 3×N matrix where N is number of VSCMGs
        h: List of wheel momentum values for each VSCMG
    """
    import numpy as np
    
    N = len(VSCMGs)
    A = np.zeros((3, N))
    h = np.zeros(N)
    
    for i, vscmg in enumerate(VSCMGs):
        # Wheel momentum: h = I_wheel * Omega
        # IW1 is the wheel inertia along spin axis
        h[i] = vscmg.IW1 * vscmg.Omega  # [kg·m²·rad/s]
        
        # Torque axis (normalize if needed)
        gtHat = np.array(vscmg.gtHat0_B).flatten()
        gtHat = gtHat / np.linalg.norm(gtHat)
        
        # Torque authority column: h_i * gtHat_i
        A[:, i] = h[i] * gtHat
    
    return A, h

def mapTorqToGimbalCmds(tau_desired, A, method='pseudo_inverse', gain_matrix=None):
    """
    Map desired body torques to gimbal torque commands.
    
    Two methods:
    1. 'pseudo_inverse': u = A^+ @ τ (least-squares solution)
    2. 'gain_matrix': u = K @ τ (if K provided)
    
    Args:
        tau_desired: 3-element array of desired body torques [Nm]
        A: Torque authority matrix (3×N)
        method: 'pseudo_inverse' or 'gain_matrix'
        gain_matrix: K matrix if using gain method
    
    Returns:
        u_gimbal: N-element array of gimbal torque commands [Nm]
    """
    import numpy as np
    
    if method == 'pseudo_inverse':
        # Least-squares solution
        A_pinv = np.linalg.pinv(A)
        u_gimbal = A_pinv @ tau_desired
    elif method == 'gain_matrix' and gain_matrix is not None:
        u_gimbal = gain_matrix @ tau_desired
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return u_gimbal

def analyzeVSCMGResponse(speedLog, gimbalAngles, gimbalRates, wheelSpeeds, 
                         guidance_df, torque_df, VSCMGs):
    """
    Analyze VSCMG gimbal and wheel response to commanded torques.
    
    Args:
        speedLog: Basilisk speed output log
        gimbalAngles: Gimbal angles from simulation [rad]
        gimbalRates: Gimbal rates from simulation [rad/s]
        wheelSpeeds: Wheel speeds from simulation [rad/s]
        guidance_df: Guidance profile DataFrame
        torque_df: Torque profile DataFrame
        VSCMGs: List of VSCMG configurations
    
    Returns:
        Analysis dict with responses
    """
    import numpy as np
    
    sim_time = speedLog.times() * 1.0E-9  # Convert to seconds
    N = len(VSCMGs)
    
    print("\n" + "="*80)
    print("VSCMG RESPONSE ANALYSIS")
    print("="*80)
    
    # Check gimbal angle changes
    print(f"\nGimbal Angle Response:")
    for i in range(N):
        gamma_init = gimbalAngles[0, i]
        gamma_final = gimbalAngles[-1, i]
        gamma_change = np.degrees(gamma_final - gamma_init)
        print(f"  VSCMG {i}: delta_gamma = {gamma_change:.4f}deg ({gamma_final-gamma_init:.6f} rad)")
    
    # Check gimbal rate peaks
    print(f"\nGimbal Rate Response:")
    for i in range(N):
        rate_max = np.max(np.abs(gimbalRates[:, i]))
        rate_max_deg = np.degrees(rate_max)
        print(f"  VSCMG {i}: Max gamma_mod = {rate_max_deg:.4f}°/s ({rate_max:.6f} rad/s)")
    
    # Check wheel speed changes
    print(f"\nWheel Speed Response:")
    for i in range(N):
        omega_init = wheelSpeeds[0, i]
        omega_final = wheelSpeeds[-1, i]
        omega_change_rpm = (omega_final - omega_init) * 60 / (2*np.pi)
        print(f"  VSCMG {i}: delta_omega = {omega_change_rpm:.2f} RPM")
    
    # Check torque capacity
    print(f"\nTorque Capacity vs. Requirement:")
    if 'tau_x' in torque_df.columns:
        tau_mag = np.sqrt(
            torque_df['tau_x'].values**2 + 
            torque_df['tau_y'].values**2 + 
            torque_df['tau_z'].values**2
        )
        tau_max = np.max(tau_mag)
        tau_mean = np.mean(tau_mag)
        print(f"  Required torque - Max: {tau_max:.6f} Nm, Mean: {tau_mean:.6f} Nm")
    
    print("="*80 + "\n")
    
    return {
        'gimbal_angles': gimbalAngles,
        'gimbal_rates': gimbalRates,
        'wheel_speeds': wheelSpeeds
    }



#==========================================

def VSCMGIntegratedTest(show_plots,useFlag,testCase):
    # The __tracebackhide__ setting influences pytest showing of tracebacks:
    # the mrp_steering_tracking() function will not be shown unless the
    # --fulltrace command line option is specified.
    __tracebackhide__ = True

    testFailCount = 0  # zero unit test result counter
    testMessages = []  # create empty list to store test log messages

    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "spacecraftBody"

    unitTaskName = "unitTask"  # arbitrary name (don't change)
    unitProcessName = "TestProcess"  # arbitrary name (don't change)

    # Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create test thread
    if testCase == 'JitterFullyCoupled' or testCase == 'JitterFullyCoupledGravity':
        dt = 0.001
        duration = 5.0
    else:
        dt = 0.001
        duration = 3.
    testProcessRate = macros.sec2nano(dt)  # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # add RW devices
    VSCMGs = []

    # CMG Pyramid Configuration
    theta_skew = 55.0 * np.pi / 180  # Skew angle [rad]
    theta_elev = 35.0 * np.pi / 180  # Elevation angle [rad]
    
    # Azimuths: 0°, 90°, 180°, 270°
    azimuths_deg = [0.0, 90.0, 180.0, 270.0]
    azimuths_rad = [az * np.pi / 180 for az in azimuths_deg]

    VSCMGs.append(defaultVSCMG())
    az0 = azimuths_rad[0]
    VSCMGs[0].gsHat0_B = [[np.sin(theta_skew) * np.cos(az0)], 
                           [np.sin(theta_skew) * np.sin(az0)], 
                           [np.cos(theta_skew)]]
    VSCMGs[0].ggHat_B = [[np.cos(theta_elev) * (-np.sin(az0))], 
                          [np.cos(theta_elev) * np.cos(az0)], 
                          [np.sin(theta_elev)]]
    VSCMGs[0].gtHat0_B = np.cross(np.array([np.sin(theta_skew) * np.cos(az0), np.sin(theta_skew) * np.sin(az0), np.cos(theta_skew)]), 
                                   np.array([np.cos(theta_elev) * (-np.sin(az0)), np.cos(theta_elev) * np.cos(az0), np.sin(theta_elev)]))
    VSCMGs[0].Omega = 2000 * macros.RPM
    VSCMGs[0].gamma = 0.
    VSCMGs[0].gammaDot = 0.0
    VSCMGs[0].rGB_B = [[0.1], [0.0], [0.05]]

    VSCMGs.append(defaultVSCMG())
    az1 = azimuths_rad[1]
    VSCMGs[1].gsHat0_B = [[np.sin(theta_skew) * np.cos(az1)], 
                           [np.sin(theta_skew) * np.sin(az1)], 
                           [np.cos(theta_skew)]]
    VSCMGs[1].ggHat_B = [[np.cos(theta_elev) * (-np.sin(az1))], 
                          [np.cos(theta_elev) * np.cos(az1)], 
                          [np.sin(theta_elev)]]
    VSCMGs[1].gtHat0_B = np.cross(np.array([np.sin(theta_skew) * np.cos(az1), np.sin(theta_skew) * np.sin(az1), np.cos(theta_skew)]), 
                                   np.array([np.cos(theta_elev) * (-np.sin(az1)), np.cos(theta_elev) * np.cos(az1), np.sin(theta_elev)]))
    VSCMGs[1].Omega =  2000 * macros.RPM
    VSCMGs[1].gamma = 0.
    VSCMGs[1].gammaDot = 0.0
    VSCMGs[1].rGB_B = [[0.0], [0.1], [0.05]]

    VSCMGs.append(defaultVSCMG())
    az2 = azimuths_rad[2]
    VSCMGs[2].gsHat0_B = [[np.sin(theta_skew) * np.cos(az2)], 
                           [np.sin(theta_skew) * np.sin(az2)], 
                           [np.cos(theta_skew)]]
    VSCMGs[2].ggHat_B = [[np.cos(theta_elev) * (-np.sin(az2))], 
                          [np.cos(theta_elev) * np.cos(az2)], 
                          [np.sin(theta_elev)]]
    VSCMGs[2].gtHat0_B = np.cross(np.array([np.sin(theta_skew) * np.cos(az2), np.sin(theta_skew) * np.sin(az2), np.cos(theta_skew)]), 
                                   np.array([np.cos(theta_elev) * (-np.sin(az2)), np.cos(theta_elev) * np.cos(az2), np.sin(theta_elev)]))
    VSCMGs[2].Omega = 2000 * macros.RPM
    VSCMGs[2].gamma = 0.
    VSCMGs[2].gammaDot = 0.0
    VSCMGs[2].rGB_B = [[-0.1], [0.0], [0.05]]

    VSCMGs.append(defaultVSCMG())
    az3 = azimuths_rad[3]
    VSCMGs[3].gsHat0_B = [[np.sin(theta_skew) * np.cos(az3)], 
                           [np.sin(theta_skew) * np.sin(az3)], 
                           [np.cos(theta_skew)]]
    VSCMGs[3].ggHat_B = [[np.cos(theta_elev) * (-np.sin(az3))], 
                          [np.cos(theta_elev) * np.cos(az3)], 
                          [np.sin(theta_elev)]]
    VSCMGs[3].gtHat0_B = np.cross(np.array([np.sin(theta_skew) * np.cos(az3), np.sin(theta_skew) * np.sin(az3), np.cos(theta_skew)]), 
                                   np.array([np.cos(theta_elev) * (-np.sin(az3)), np.cos(theta_elev) * np.cos(az3), np.sin(theta_elev)]))
    VSCMGs[3].Omega = 2000 * macros.RPM          # Wheel speed
    VSCMGs[3].gamma = 0.                         # Gimbal angle
    VSCMGs[3].gammaDot = 0.0                     # Gimbal rate
    VSCMGs[3].rGB_B = [[0.0], [-0.1], [0.05]]   # Position in body frame

    if testCase == 'BalancedWheels':
        VSCMGModel = 0
    elif testCase == 'JitterSimple':
        VSCMGModel = 1
    elif testCase == 'JitterFullyCoupled' or testCase == 'JitterFullyCoupledGravity':
        VSCMGModel = 2

    for VSCMG in VSCMGs:
        VSCMG.VSCMGModel = VSCMGModel

    if testCase == 'JitterFullyCoupled':
        VSCMGs = [VSCMGs[0],VSCMGs[1],VSCMGs[2],VSCMGs[3]]

    N = len(VSCMGs)

    # create VSCMG object container and tie to spacecraft object
    rwStateEffector = vscmgStateEffector.VSCMGStateEffector()
    rwStateEffector.ModelTag = "VSCMGs"
    for item in VSCMGs:
        rwStateEffector.AddVSCMG(item)
    scObject.addStateEffector(rwStateEffector)

    # ===== FSW CONTROLLER SETUP =====
    from Basilisk.fswAlgorithms import mrpFeedback
    from Basilisk.fswAlgorithms import inertial3D
    from Basilisk.fswAlgorithms import attTrackingError
    from Basilisk.simulation import simpleNav

    # Navigation module
    sNavObject = simpleNav.SimpleNav()
    sNavObject.ModelTag = "SimpleNav"
    sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)

    # Reference attitude (desired: 30° roll maneuver)
    inertialObj = inertial3D.inertial3D()
    inertialObj.ModelTag = "inertial3D"
    sigma_desired = roll_to_mrp(np.radians(30.0))
    inertialObj.sigma_R0N = sigma_desired.tolist()

    # Use a desired pitch/yaw that cancels coupling:
    # sigma_des_roll = roll_to_mrp(np.radians(30.0))
    # sigma_des_pitch = roll_to_mrp(np.radians(0.0))  # Adjusted to suppress coupling
    # sigma_des_yaw = roll_to_mrp(np.radians(0.0))   # Adjusted to suppress coupling
    # inertialObj.sigma_R0N = [sigma_des_roll[0], sigma_des_pitch[1], sigma_des_yaw[2]]

    # Tracking error
    attError = attTrackingError.attTrackingError()
    attError.ModelTag = "attError"
    attError.attNavInMsg.subscribeTo(sNavObject.attOutMsg)
    attError.attRefInMsg.subscribeTo(inertialObj.attRefOutMsg)

    # MRP Feedback Controller
    mrpControl = mrpFeedback.mrpFeedback()
    mrpControl.ModelTag = "mrpFeedback"
    mrpControl.K = 1.5      # Proportional gain [Nm/(MRP unit)]
    mrpControl.P = 3.75     # Rate feedback gain [Nm·s/rad]
    mrpControl.Ki = -1.0    # Integral gain (negative = off)

    # Vehicle config
    configData = messaging.VehicleConfigMsgPayload()
    configData.ISCPntB_B = [55.61, 0.0, 0.0, 0.0, 105.93, 0.0, 0.0, 0.0, 114.81]
    configMsg = messaging.VehicleConfigMsg().write(configData)
    mrpControl.vehConfigInMsg.subscribeTo(configMsg)
    mrpControl.guidInMsg.subscribeTo(attError.attGuidOutMsg)

    # Custom gimbal command mapper
    class VSCMGCommandMapper(SysModel):
        def __init__(self, VSCMGs, N):
            super().__init__()
            self.ModelTag = "VSCMGCommandMapper"
            self.VSCMGs = VSCMGs  
            self.N = N
            self.cmdTorqueInMsg = None
            self.cmdArrayMsg = messaging.VSCMGArrayTorqueMsg()
            self.speedLog = None  

        def UpdateState(self, CurrentSimNanos):
            if self.cmdTorqueInMsg is not None:
                # ===== REBUILD A EVERY TIMESTEP =====
                A, h = buildTorqueAuthorityMatrix(self.VSCMGs)  # ← Recompute with current Ω
                
                cmdTorque = self.cmdTorqueInMsg.read()
                tau_desired = np.array([cmdTorque.torqueRequestBody[0],
                                    cmdTorque.torqueRequestBody[1],
                                    cmdTorque.torqueRequestBody[2]])
                
                u_gimbal = mapTorqToGimbalCmds(tau_desired, A, method='pseudo_inverse')
                
                cmdArray = messaging.VSCMGArrayTorqueMsgPayload()
                cmdArray.wheelTorque = [0.0] * self.N
                cmdArray.gimbalTorque = u_gimbal.tolist()
                self.cmdArrayMsg.write(cmdArray)


    # Setup mapper
    cmdMapper = VSCMGCommandMapper(VSCMGs, N)
    cmdMapper.cmdTorqueInMsg = mrpControl.cmdTorqueOutMsg
    rwStateEffector.cmdsInMsg.subscribeTo(cmdMapper.cmdArrayMsg)

    # Add modules to task
    unitTestSim.AddModelToTask(unitTaskName, sNavObject)
    unitTestSim.AddModelToTask(unitTaskName, inertialObj)
    unitTestSim.AddModelToTask(unitTaskName, attError)
    unitTestSim.AddModelToTask(unitTaskName, mrpControl)
    unitTestSim.AddModelToTask(unitTaskName, cmdMapper)

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, rwStateEffector)
    unitTestSim.AddModelToTask(unitTaskName, scObject)

    if testCase != 'JitterFullyCoupled':
        unitTestSim.earthGravBody = gravityEffector.GravBodyData()
        unitTestSim.earthGravBody.planetName = "earth_planet_data"
        unitTestSim.earthGravBody.mu = 0.3986004415E+15 # meters!
        unitTestSim.earthGravBody.isCentralBody = True

        scObject.gravField.gravBodies = spacecraft.GravBodyVector([unitTestSim.earthGravBody])

    scObject.hub.mHub = 212.54 #LB value
    scObject.hub.r_BcB_B = [[-0.01], [-0.09], [0.51]] #LB value
    scObject.hub.IHubPntBc_B = [[55.61, 0.0, 0.0], [0.0, 105.93, 0.0], [0.0, 0.0, 114.81]] #LB value

    if testCase == 'JitterFullyCoupled':
        scObject.hub.r_CN_NInit = [[0.1],	[-0.2],	[0.3]]
        scObject.hub.v_CN_NInit = [[-0.4],	[0.5],	[-0.8]]
    else:
        scObject.hub.r_CN_NInit = [[-4020338.690396649],	[7490566.741852513],	[5248299.211589362]]
        scObject.hub.v_CN_NInit = [[-5199.77710904224],	[-3436.681645356935],	[1041.576797498721]]
    scObject.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]
    scObject.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]

    # log data
    dataLog = scObject.scStateOutMsg.recorder()
    speedLog = rwStateEffector.speedOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)
    unitTestSim.AddModelToTask(unitTaskName, speedLog)

    scObjectLog = scObject.logger(["totOrbAngMomPntN_N", "totRotAngMomPntC_N", "totOrbEnergy", "totRotEnergy"])
    unitTestSim.AddModelToTask(unitTaskName, scObjectLog)

    unitTestSim.ConfigureStopTime(macros.sec2nano(duration))

    unitTestSim.InitializeSimulation()

    posRef = scObject.dynManager.getStateObject(scObject.hub.nameOfHubPosition)
    sigmaRef = scObject.dynManager.getStateObject(scObject.hub.nameOfHubSigma)

    unitTestSim.ExecuteSimulation()
    
    orbAngMom_N = simHelpers.addTimeColumn(scObjectLog.times(), scObjectLog.totOrbAngMomPntN_N)
    rotAngMom_N = simHelpers.addTimeColumn(scObjectLog.times(), scObjectLog.totRotAngMomPntC_N)
    rotEnergy = simHelpers.addTimeColumn(scObjectLog.times(), scObjectLog.totRotEnergy)
    orbEnergy = simHelpers.addTimeColumn(scObjectLog.times(), scObjectLog.totOrbEnergy)

    wheelSpeeds = speedLog.wheelSpeeds
    gimbalAngles = speedLog.gimbalAngles
    gimbalRates = speedLog.gimbalRates
    sigmaData = dataLog.sigma_BN
    omegaData = dataLog.omega_BN_B

    # Convert gimbal angles to degrees
    gimbalAngles_deg = np.degrees(gimbalAngles)

    # Convert gimbal rates to rad/s (already in rad/s) and degrees/s
    gimbalRates_rad = gimbalRates
    gimbalRates_deg = np.degrees(gimbalRates)

    # Convert sigma to Quaternions and Euler angles
    quaternions = np.zeros((len(sigmaData), 4))
    eulerAngles = np.zeros((len(sigmaData), 3))
    eulerAngles_deg = np.zeros((len(sigmaData), 3))

    for i in range(len(sigmaData)):
        quaternions[i] = mrp2Quaternion(sigmaData[i])
        eulerAngles[i] = mrp2EulerAngles(sigmaData[i])
        eulerAngles_deg[i] = np.degrees(eulerAngles[i])

    # Print conversions to console
    print("\n" + "="*80)
    print("ATTITUDE CONVERSIONS AND DATA LOGGING")
    print("="*80)
    print(f"\nInitial State:")
    print(f"  MRP (sigma): {sigmaData[0]}")
    print(f"  Quaternion (q0, q1, q2, q3): {quaternions[0]}")
    print(f"  Euler Angles [rad]: {eulerAngles[0]}")
    print(f"  Euler Angles [deg]: {eulerAngles_deg[0]}")

    print(f"\nFinal State:")
    print(f"  MRP (sigma): {sigmaData[-1]}")
    print(f"  Quaternion (q0, q1, q2, q3): {quaternions[-1]}")
    print(f"  Euler Angles [rad]: {eulerAngles[-1]}")
    print(f"  Euler Angles [deg]: {eulerAngles_deg[-1]}")

    print(f"\nGimbal Angles (Sample at t=0):")
    for i in range(N):
        print(f"  VSCMG {i}: {gimbalAngles[0,i]:.6f} rad = {gimbalAngles_deg[0,i]:.6f} deg")

    print(f"\nGimbal Rates (Sample at t=0):")
    for i in range(N):
        print(f"  VSCMG {i}: {gimbalRates_rad[0,i]:.6f} rad/s = {gimbalRates_deg[0,i]:.6f} deg/s")
    print("="*80 + "\n")

    dataPos = posRef.getState()
    dataSigma = sigmaRef.getState()
    dataPos = [[dataPos[0][0], dataPos[1][0], dataPos[2][0]]]
    dataSigma = [[dataSigma[0][0], dataSigma[1][0], dataSigma[2][0]]]

    if testCase == 'BalancedWheels':
        truePos = [
            [-4025537.664298894, 7487128.570444949, 5249339.643828076]
        ]

        trueSigma = [
            [0.0185829763256, 0.00212563436704, -0.00118728497031]
        ]

    elif testCase == 'JitterSimple':
        truePos = [
            [-4025537.659558947, 7487128.570662447, 5249339.653774626]
        ]

        trueSigma = [
            [0.018774477186285467, 0.0018376842577357564, -0.00023633044221463834]
        ]

    elif testCase == 'JitterFullyCoupled':
        truePos = [
            [0.0970572658434, -0.195562924079, 0.191874379545]
        ]

        trueSigma = [
            [0.000201909373901, 2.9217809421e-05, 4.00231302121e-06]
        ]
    elif testCase == 'JitterFullyCoupledGravity':
        truePos = [
            [-4020390.68802, 7490532.37502, 5248309.52745]
        ]

        trueSigma = [
            [0.000201662012765, 2.92123940252e-05, 4.15606551702e-06]
        ]



    initialOrbAngMom_N = [
                [orbAngMom_N[0,1], orbAngMom_N[0,2], orbAngMom_N[0,3]]
                ]

    finalOrbAngMom = [
                [orbAngMom_N[-1,1], orbAngMom_N[-1,2], orbAngMom_N[-1,3]]
                 ]

    initialRotAngMom_N = [
                [rotAngMom_N[0,1], rotAngMom_N[0,2], rotAngMom_N[0,3]]
                ]

    finalRotAngMom = [
                [rotAngMom_N[-1,1], rotAngMom_N[-1,2], rotAngMom_N[-1,3]]
                 ]

    initialOrbEnergy = [
                [orbEnergy[0,1]]
                ]

    finalOrbEnergy = [
                [orbEnergy[-1,1]]
                 ]

    initialRotEnergy = [
                [rotEnergy[0,1]]
                ]

    finalRotEnergy = [
                [rotEnergy[-1,1]]
                 ]
    plt.close("all")

    plt.figure()
    for i in range(0,N):
        plt.subplot(4,1,i+1)
        plt.plot(speedLog.times()*1.0E-9, wheelSpeeds[:,i] / (2.0 * math.pi) * 60, label='RWA' + str(i))
        plt.xlabel('Time (s)')
        plt.ylabel(r'RW' + str(i) + r' $\Omega$ [RPM]')
    plt.suptitle('Wheel Speeds [RPM]', fontsize=14, fontweight='bold', y=1.00)

    plt.figure()
    for i in range(0,N):
        plt.subplot(4,1,i+1)
        plt.plot(speedLog.times()*1.0E-9, gimbalAngles[:,i], label=str(i))
        plt.xlabel('Time (s)')
        plt.ylabel(r'$\gamma_'+str(i)+'$ [rad]')
    plt.suptitle('Gimbal Angles [rad]', fontsize=14, fontweight='bold', y=1.00)

    # Gimbal Angles in Degrees
    plt.figure()
    for i in range(0,N):
        plt.subplot(4,1,i+1)
        plt.plot(speedLog.times()*1.0E-9, gimbalAngles_deg[:,i], label=str(i))
        plt.xlabel('Time (s)')
        plt.ylabel(r'$\gamma_'+str(i)+'$ [deg]')
    plt.suptitle('Gimbal Angles [deg]', fontsize=14, fontweight='bold', y=1.00)

    plt.figure()
    for i in range(0,N):
        plt.subplot(4,1,i+1)
        plt.plot(speedLog.times()*1.0E-9, gimbalRates[:,i] * 180/np.pi, label=str(i))
        plt.xlabel('Time (s)')
        plt.ylabel(r'$\dot{\gamma}_'+str(i)+'$ [deg/s]')
    plt.suptitle('Gimbal Rates [deg/s]', fontsize=14, fontweight='bold', y=1.00)

    # Gimbal Rates in rad/s
    plt.figure()
    for i in range(0,N):
        plt.subplot(4,1,i+1)
        plt.plot(speedLog.times()*1.0E-9, gimbalRates_rad[:,i], label=str(i))
        plt.xlabel('Time (s)')
        plt.ylabel(r'$\dot{\gamma}_'+str(i)+'$ [rad/s]')
    plt.suptitle('Gimbal Rates [rad/s]', fontsize=14, fontweight='bold', y=1.00)

    plt.figure()
    for i in range(0,3):
        plt.subplot(4,1,i+1)
        plt.plot(dataLog.times()*1.0E-9, sigmaData[:,i], label='MRP' + str(i))
        plt.xlabel('Time (s)')
        plt.ylabel(r'$\sigma_'+str(i)+'$ [-]')
    plt.suptitle('Attitude (MRP Components) [-]', fontsize=14, fontweight='bold', y=1.00)

    plt.figure()
    for i in range(0,3):
        plt.subplot(4,1,i+1)
        plt.plot(dataLog.times()*1.0E-9, omegaData[:,i] * 180/math.pi, label='omega' + str(i))
        plt.xlabel('Time (s)')
        plt.ylabel(r'$\omega_'+str(i)+'$ [deg/s]')
    plt.suptitle('Angular Velocity [deg/s]', fontsize=14, fontweight='bold', y=1.00)

    # Body Attitude - Quaternions [q0, q1, q2, q3]
    plt.figure()
    labels = ['q0 (scalar)', 'q1 (i)', 'q2 (j)', 'q3 (k)']
    for i in range(0,4):
        plt.subplot(4,1,i+1)
        plt.plot(dataLog.times()*1.0E-9, quaternions[:,i], label=labels[i])
        plt.xlabel('Time (s)')
        plt.ylabel(labels[i])
    plt.suptitle('Body Attitude - Quaternions [-]', fontsize=14, fontweight='bold', y=1.00)

    # Body Attitude - Euler Angles [rad]
    plt.figure()
    euler_labels_rad = [r'Roll $\phi$ [rad]', r'Pitch $\theta$ [rad]', r'Yaw $\psi$ [rad]']
    for i in range(0,3):
        plt.subplot(4,1,i+1)
        plt.plot(dataLog.times()*1.0E-9, eulerAngles[:,i], label=euler_labels_rad[i])
        plt.xlabel('Time (s)')
        plt.ylabel(euler_labels_rad[i])
    plt.suptitle('Body Attitude - Euler Angles [rad]', fontsize=14, fontweight='bold', y=1.00)

    # Body Attitude - Euler Angles [deg]
    plt.figure()
    euler_labels_deg = [r'Roll $\phi$ [deg]', r'Pitch $\theta$ [deg]', r'Yaw $\psi$ [deg]']
    for i in range(0,3):
        plt.subplot(4,1,i+1)
        plt.plot(dataLog.times()*1.0E-9, eulerAngles_deg[:,i], label=euler_labels_deg[i])
        plt.xlabel('Time (s)')
        plt.ylabel(euler_labels_deg[i])
    plt.suptitle('Body Attitude - Euler Angles [deg]', fontsize=14, fontweight='bold', y=1.00)



    if show_plots == True:
        plt.show()
        plt.close('all')

    accuracy = 1e-7
    for i in range(0,len(truePos)):
        # check a vector values
        if not unitTestSupport.isArrayEqualRelative(dataPos[i],truePos[i],3,accuracy):
            testFailCount += 1
            testMessages.append("FAILED: VSCMG Integrated Test failed pos unit test")

    for i in range(0,len(trueSigma)):
        # check a vector values
        if not unitTestSupport.isArrayEqualRelative(dataSigma[i],trueSigma[i],3,accuracy):
            testFailCount += 1
            testMessages.append("FAILED: VSCMG Integrated Test failed attitude unit test")

    accuracy = 1e-10
    if testCase == 'BalancedWheels' or testCase == 'JitterFullyCoupled':

        for i in range(0,len(initialOrbAngMom_N)):
            # check a vector values
            if not unitTestSupport.isArrayEqualRelative(finalOrbAngMom[i],initialOrbAngMom_N[i],3,accuracy):
                testFailCount += 1
                testMessages.append("FAILED: VSCMG Integrated Test failed orbital angular momentum unit test")

        for i in range(0,len(initialRotAngMom_N)):
            # check a vector values
            if not unitTestSupport.isArrayEqualRelative(finalRotAngMom[i],initialRotAngMom_N[i],3,accuracy):
                testFailCount += 1
                testMessages.append("FAILED: VSCMG Integrated Test failed rotational angular momentum unit test")

        for i in range(0, len(initialOrbEnergy)):
            # check a vector values
            if not unitTestSupport.isArrayEqualRelative(finalOrbEnergy[i], initialOrbEnergy[i], 1, accuracy):
                testFailCount += 1
                testMessages.append("FAILED: VSCMG Integrated Test failed orbital energy unit test")

        for i in range(0, len(initialRotEnergy)):
            # check a vector values
            if not unitTestSupport.isArrayEqualRelative(finalRotEnergy[i], initialRotEnergy[i], 1, accuracy):
                testFailCount += 1
                testMessages.append("FAILED: VSCMG Integrated Test failed rot energy unit test")

    # print out success message if no errors were found
    if  testFailCount == 0:
        print("PASSED ")
        colorText = 'ForestGreen'
        passedText = r'\textcolor{' + colorText + '}{' + "PASSED" + '}'
        # Write some snippets for AutoTex
        snippetName = testCase + 'PassFail'
        simHelpers.writeTeXSnippet(snippetName, passedText, path)
    elif testFailCount > 0:
        colorText = 'Red'
        passedText = r'\textcolor{' + colorText + '}{' + "FAILED" + '}'
        # Write some snippets for AutoTex
        snippetName = testCase + 'PassFail'
        simHelpers.writeTeXSnippet(snippetName, passedText, path)

    if testFailCount == 0:
        print("PASSED: " + " VSCMG Integrated Sim Test")
    else:
        print(testMessages)

    assert testFailCount < 1, testMessages

    # return fail count and join into a single string all messages in the list
    # testMessage
    return [testFailCount, ''.join(testMessages)]

if __name__ == "__main__":
    VSCMGIntegratedTest(True,False,'JitterFullyCoupled')
