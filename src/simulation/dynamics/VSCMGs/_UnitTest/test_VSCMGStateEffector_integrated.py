
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
from Basilisk.utilities import RigidBodyKinematics as rbk
import numpy as np

import yaml
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.yaml')

# Load the YAML file
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

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
    VSCMG.Omega_max = 8000. * macros.RPM
    VSCMG.gammaDot_max = -1
    VSCMG.IW1 = 4./VSCMG.Omega_max 
    VSCMG.IW2 = 0.5*VSCMG.IW1 
    VSCMG.IW3 = 0.5*VSCMG.IW1 
    VSCMG.IG1 = 1.037e-3
    VSCMG.IG2 = 1.037e-3
    VSCMG.IG3 = 1.037e-3
    VSCMG.U_s = config['vscmg_static_imbalance']  #4.8e-06 * 1e4
    VSCMG.U_d = config['vscmg_dynamic_imbalance']  #1.54e-06 * 1e4
    VSCMG.l = 0.0 #0.01
    VSCMG.L = 0.0 #0.1
    VSCMG.rGcG_G = [[0.0],[0.0],[0.0]]
    VSCMG.massW = 1.6
    VSCMG.massG = 1.0
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
        dt = config['simulation_step_size']
        duration = config['simulation_time']
    else:
        dt = config['simulation_step_size']
        duration = config['simulation_time']
    testProcessRate = macros.sec2nano(dt)  # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # add RW devices
    VSCMGs = []

    ang = config['skew_angle']
    az_angles = [0,90,180,270]

    VSCMGs.append(defaultVSCMG())
    VSCMGs[0].gsHat0_B = [[np.cos(np.deg2rad(ang))*np.cos(np.deg2rad(az_angles[0]))], [np.cos(np.deg2rad(ang))*np.sin(np.deg2rad(az_angles[0]))], [np.sin(np.deg2rad(ang))]]
    VSCMGs[0].ggHat_B = [[-np.cos(np.deg2rad(90-ang))*np.cos(np.deg2rad(az_angles[0]))], [np.cos(np.deg2rad(90-ang))*np.sin(np.deg2rad(az_angles[0]))], [np.sin(np.deg2rad(90-ang))]]
    gs0 = np.array([VSCMGs[0].gsHat0_B[0][0], VSCMGs[0].gsHat0_B[1][0], VSCMGs[0].gsHat0_B[2][0]])
    gg0 = np.array([VSCMGs[0].ggHat_B[0][0], VSCMGs[0].ggHat_B[1][0], VSCMGs[0].ggHat_B[2][0]])
    gt0 = np.cross(gg0, gs0)
    gt0 = gt0 / np.linalg.norm(gt0)
    VSCMGs[0].gtHat0_B = [[gt0[0]], [gt0[1]], [gt0[2]]]
    VSCMGs[0].Omega = config['initial_vscmg_wheel_speeds_rpm'][0] * macros.RPM
    VSCMGs[0].gamma = config['initial_vscmg_gimbal_angles'][0] * np.pi/180
    VSCMGs[0].gammaDot = config['initial_vscmg_gimbal_rates'][0] * np.pi/180
    VSCMGs[0].rGB_B = [[0.4], [0.0], [0.0]]

    VSCMGs.append(defaultVSCMG())
    VSCMGs[1].gsHat0_B = [[np.cos(np.deg2rad(ang))*np.cos(np.deg2rad(az_angles[1]))], [np.cos(np.deg2rad(ang))*np.sin(np.deg2rad(az_angles[1]))], [np.sin(np.deg2rad(ang))]]
    VSCMGs[1].ggHat_B = [[np.cos(np.deg2rad(90-ang))*np.cos(np.deg2rad(az_angles[1]))], [-np.cos(np.deg2rad(90-ang))*np.sin(np.deg2rad(az_angles[1]))], [np.sin(np.deg2rad(90-ang))]]
    gs1 = np.array([VSCMGs[1].gsHat0_B[0][0], VSCMGs[1].gsHat0_B[1][0], VSCMGs[1].gsHat0_B[2][0]])
    gg1 = np.array([VSCMGs[1].ggHat_B[0][0], VSCMGs[1].ggHat_B[1][0], VSCMGs[1].ggHat_B[2][0]])
    gt1 = np.cross(gg1, gs1)
    gt1 = gt1 / np.linalg.norm(gt1)
    VSCMGs[1].gtHat0_B = [[gt1[0]], [gt1[1]], [gt1[2]]]
    VSCMGs[1].Omega = config['initial_vscmg_wheel_speeds_rpm'][1] * macros.RPM
    VSCMGs[1].gamma = config['initial_vscmg_gimbal_angles'][1] * np.pi/180
    VSCMGs[1].gammaDot = config['initial_vscmg_gimbal_rates'][1] * np.pi/180
    VSCMGs[1].rGB_B = [[0.0], [0.4], [0.0]]

    VSCMGs.append(defaultVSCMG())
    VSCMGs[2].gsHat0_B = [[np.cos(np.deg2rad(ang))*np.cos(np.deg2rad(az_angles[2]))], [np.cos(np.deg2rad(ang))*np.sin(np.deg2rad(az_angles[2]))], [np.sin(np.deg2rad(ang))]]
    VSCMGs[2].ggHat_B = [[-np.cos(np.deg2rad(90-ang))*np.cos(np.deg2rad(az_angles[2]))], [np.cos(np.deg2rad(90-ang))*np.sin(np.deg2rad(az_angles[2]))], [np.sin(np.deg2rad(90-ang))]]
    gs2 = np.array([VSCMGs[2].gsHat0_B[0][0], VSCMGs[2].gsHat0_B[1][0], VSCMGs[2].gsHat0_B[2][0]])
    gg2 = np.array([VSCMGs[2].ggHat_B[0][0], VSCMGs[2].ggHat_B[1][0], VSCMGs[2].ggHat_B[2][0]])
    gt2 = np.cross(gg2, gs2)
    gt2 = gt2 / np.linalg.norm(gt2)
    VSCMGs[2].gtHat0_B = [[gt2[0]], [gt2[1]], [gt2[2]]]
    VSCMGs[2].Omega = config['initial_vscmg_wheel_speeds_rpm'][2] * macros.RPM
    VSCMGs[2].gamma = config['initial_vscmg_gimbal_angles'][2] * np.pi/180
    VSCMGs[2].gammaDot = config['initial_vscmg_gimbal_rates'][2] * np.pi/180
    VSCMGs[2].rGB_B = [[-0.4], [0.0], [0.0]]

    VSCMGs.append(defaultVSCMG())
    VSCMGs[3].gsHat0_B = [[np.cos(np.deg2rad(ang))*np.cos(np.deg2rad(az_angles[3]))], [np.cos(np.deg2rad(ang))*np.sin(np.deg2rad(az_angles[3]))], [np.sin(np.deg2rad(ang))]]  # Spin axis direction
    VSCMGs[3].ggHat_B = [[np.cos(np.deg2rad(90-ang))*np.cos(np.deg2rad(az_angles[3]))], [-np.cos(np.deg2rad(90-ang))*np.sin(np.deg2rad(az_angles[3]))], [np.sin(np.deg2rad(90-ang))]] # Gimbal axis
    gs3 = np.array([VSCMGs[3].gsHat0_B[0][0], VSCMGs[3].gsHat0_B[1][0], VSCMGs[3].gsHat0_B[2][0]])
    gg3 = np.array([VSCMGs[3].ggHat_B[0][0], VSCMGs[3].ggHat_B[1][0], VSCMGs[3].ggHat_B[2][0]])
    gt3 = np.cross(gg3, gs3)
    gt3 = gt3 / np.linalg.norm(gt3)
    VSCMGs[3].gtHat0_B = [[gt3[0]], [gt3[1]], [gt3[2]]]
    VSCMGs[3].Omega = config['initial_vscmg_wheel_speeds_rpm'][3] * macros.RPM
    VSCMGs[3].gamma = config['initial_vscmg_gimbal_angles'][3] * np.pi/180
    VSCMGs[3].gammaDot = config['initial_vscmg_gimbal_rates'][3] * np.pi/180
    VSCMGs[3].rGB_B = [[0.0], [-0.4], [0.0]]  # Position in body frame

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

    # set RW torque command
    cmdArray = messaging.VSCMGArrayTorqueMsgPayload()
    if testCase == 'BalancedWheels' or testCase == 'JitterFullyCoupled':
        cmdArray.wheelTorque = config['wheel_torque']
        cmdArray.gimbalTorque = config['gimbal_torque']
    else:
        # cmdArray.wheelTorque = [0.001, 0.005, -0.009] # [Nm]
        # cmdArray.gimbalTorque = [0.008, -0.0015, -0.006] # [Nm]
        cmdArray.wheelTorque = config['wheel_torque'] # [Nm]
        cmdArray.gimbalTorque = config['gimbal_torque'] # [Nm]
    cmdMsg = messaging.VSCMGArrayTorqueMsg().write(cmdArray)
    rwStateEffector.cmdsInMsg.subscribeTo(cmdMsg)

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, rwStateEffector)
    unitTestSim.AddModelToTask(unitTaskName, scObject)

    if testCase != 'JitterFullyCoupled':
        unitTestSim.earthGravBody = gravityEffector.GravBodyData()
        unitTestSim.earthGravBody.planetName = "earth_planet_data"
        unitTestSim.earthGravBody.mu = 0.3986004415E+15 # meters!
        unitTestSim.earthGravBody.isCentralBody = True

        scObject.gravField.gravBodies = spacecraft.GravBodyVector([unitTestSim.earthGravBody])


    if testCase == 'JitterFullyCoupled':
        scObject.hub.r_CN_NInit = [[0.1],	[-0.2],	[0.3]]
        scObject.hub.v_CN_NInit = [[-0.4],	[0.5],	[-0.8]]
    else:
        scObject.hub.r_CN_NInit = [[-4020338.690396649],	[7490566.741852513],	[5248299.211589362]]
        scObject.hub.v_CN_NInit = [[-5199.77710904224],	[-3436.681645356935],	[1041.576797498721]]

    # Initial spacecraft attitude in Euler angles (321 sequence) 
    euler_angles_321 = np.deg2rad(config['initial_attitude'])  # Example Euler angles in radians
    mrp = rbk.euler3212MRP(euler_angles_321)  # Convert Euler angles to MRP

    # set the initial spacecraft attitude and angular velocity states
    scObject.hub.sigma_BNInit = [mrp[0], mrp[1], mrp[2]]
    scObject.hub.omega_BN_BInit = [config['initial_angular_velocity'][0], config['initial_angular_velocity'][1], config['initial_angular_velocity'][2]]

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
    sigmaData = dataLog.sigma_BN #plotting
    omegaData = dataLog.omega_BN_B #plotting

    dataPos = posRef.getState()
    dataSigma = sigmaRef.getState() #validation
    dataPos = [[dataPos[0][0], dataPos[1][0], dataPos[2][0]]]
    dataSigma = [[dataSigma[0][0], dataSigma[1][0], dataSigma[2][0]]] 

    # Convert all MRP data to Euler angles (321 sequence)  
    eulerAngles = np.array(np.rad2deg([rbk.MRP2Euler321(sigma) for sigma in sigmaData]))

    # Convert time-series MRP to quaternions
    quaternions = np.array([rbk.MRP2EP(sigma) for sigma in sigmaData])

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
        plt.ylabel(r'RW' + str(i) + r' $\Omega$ (RPM)')

    plt.figure()
    for i in range(0,N):
        plt.subplot(4,1,i+1)
        plt.plot(speedLog.times()*1.0E-9, gimbalAngles[:,i], label=str(i))
        plt.xlabel('Time (s)')
        plt.ylabel(r'$\gamma_'+str(i)+'$ (rad)')

    plt.figure()
    for i in range(0,N):
        plt.subplot(4,1,i+1)
        plt.plot(speedLog.times()*1.0E-9, gimbalRates[:,i] * 180/np.pi, label=str(i))
        plt.xlabel('Time (s)')
        plt.ylabel(r'$\dot{\gamma}_'+str(i)+'$ (d/s)')

    plt.figure()
    for i in range(0,3):
        plt.subplot(4,1,i+1)
        plt.plot(dataLog.times()*1.0E-9, sigmaData[:,i], label='MRP' + str(i))
        plt.xlabel('Time (s)')
        plt.ylabel(r'MRP b' + str(i))

    plt.figure()
    for i in range(0,3):
        plt.subplot(4,1,i+1)
        plt.plot(dataLog.times()*1.0E-9, omegaData[:,i] * 180/math.pi, label='omega' + str(i))
        plt.xlabel('Time (s)')
        plt.ylabel(r'b' + str(i) + r' $\omega$ (d/s)')

    plt.figure()
    plt.plot(dataLog.times()*1e-9, eulerAngles)
    plt.xlabel('Time (s)')
    plt.ylabel('Euler Angles (degrees)')
    plt.legend(['Yaw', 'Pitch', 'Roll'])
    plt.title('Euler Angles (321 sequence)')
    plt.grid()

    plt.figure()
    plt.plot(dataLog.times()*1e-9, quaternions)
    plt.xlabel('Time (s)')
    plt.ylabel('Quaternion Value')
    plt.legend(['q0', 'q1', 'q2', 'q3'])
    plt.title('Quaternions')
    plt.grid()

    # Plot rotational angular momentum
    plt.figure()
    plt.plot(scObjectLog.times()*1.0E-9, rotAngMom_N[:,1], label='L_x')
    plt.plot(scObjectLog.times()*1.0E-9, rotAngMom_N[:,2], label='L_y')
    plt.plot(scObjectLog.times()*1.0E-9, rotAngMom_N[:,3], label='L_z')
    plt.xlabel('Time (s)')
    plt.ylabel('Rotational Angular Momentum (kg*m^2/s)')
    plt.title('Total Rotational Angular Momentum')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot rotational kinetic energy
    plt.figure()
    plt.plot(scObjectLog.times()*1.0E-9, rotEnergy[:,1])
    plt.xlabel('Time (s)')
    plt.ylabel('Rotational Kinetic Energy (J)')
    plt.title('Total Rotational Energy')
    plt.grid(True, alpha=0.3)

    if testCase != 'BalancedWheels' and testCase != 'JitterFullyCoupledGravity':
        istart = int(.01/dt)
        sigmaDataCut = sigmaData#sigmaData[istart:,:]
        thetaData = np.empty([len(sigmaDataCut[:,0]),2])
        thetaData[:,0] = sigmaDataCut[:,0]
        for i in range(0,len(thetaData[:,0])):
            thetaData[i,1] = 4*np.arctan(np.linalg.norm(sigmaDataCut[i,1:]))

        if testCase == 'JitterSimple':
            fitOrd = 11
        else:
            fitOrd = 9

        thetaFit = np.empty([len(sigmaDataCut[:,0]),2])
        thetaFit[:,0] = thetaData[:,0]
        p = np.polyfit(thetaData[:,0]*1e-9,thetaData[:,1],fitOrd)
        thetaFit[:,1] = np.polyval(p,thetaFit[:,0]*1e-9)

        plt.figure()
        plt.plot(thetaData[:,0]*1e-9, thetaData[:,1]-thetaFit[:,1])
        plt.title("Principle Angle Fit")
        plt.xlabel('Time (s)')
        plt.ylabel(r'$\theta$ (deg)')

        [frq,Y] = computeFFT(thetaData[:,1]-thetaFit[:,1],dt)
        peakIdxs = findPeaks(Y,N)
        wheelSpeeds_data = np.array(frq[peakIdxs])*60.
        wheelSpeeds_true = np.sort(abs(np.array([VSCMG.Omega/macros.RPM for VSCMG in VSCMGs])))

        fig, ax = plt.subplots(2,1)
        ax[0].plot(thetaFit[:,0]*1e-9,thetaData[:,1]-thetaFit[:,1])
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Amplitude')
        ax[1].plot(frq,abs(Y),'r')
        ax[1].set_xlabel('Freq (Hz)')
        ax[1].set_ylabel('Magnitude')
        ax[1].plot(frq[peakIdxs],Y[peakIdxs],'bo')
        plt.xlim((0,VSCMGs[0].Omega_max/macros.RPM/60.))

        plt.figure()
        plt.plot(thetaData[:,0]*1e-9, thetaData[:,1])
        plt.title("Principle Angle")
        plt.xlabel('Time (s)')
        plt.ylabel(r'$\theta$ (deg)')

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

    if testCase == 'JitterSimple':
        for i in range(N):
            # check a vector values
            if not abs(wheelSpeeds_data[i] - wheelSpeeds_true[i]) / wheelSpeeds_true[i] < .09:
                testFailCount += 1
                testMessages.append("FAILED: VSCMG Integrated Test failed jitter unit test")

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
    VSCMGIntegratedTest(True,True,config['test_case'])  # show_plots, useFlag, testCase
