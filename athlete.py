import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from segments import Trunk, Thigh, Shank, Arm, Bar


class Athlete(object):
    
    def __init__(self, height, weight, bar_weight, origin=(0,0)):
        self.height = height
        self.weight = weight

        heightFoot = self.height * .0425

        self.origin = np.array(origin)

        self.ankle = self.origin + np.array([0, heightFoot])
        self.shank = Shank(height, angle=90, mass=weight)
        self.thigh = Thigh(height, angle=90, mass=weight)
        self.trunk = Trunk(height, angle=90, mass=weight)
        self.arm = Arm(height, angle=-90, mass=weight)
        self.bar = Bar(100)

    def calculateKinetics(self):
        #Joint Force Calculations
        g = 9.81
        mass_bar = self.bar.mass
        BAR_ACCELx = self.bar.accel_x
        BAR_ACCELy = self.bar.accel_y

        F_BARx = mass_bar * BAR_ACCELx
        F_BARy = mass_bar * (g + BAR_ACCELy)

        F_HANDx = -F_BARx
        F_HANDy = -F_BARy

        #self.arm.distal_force = (F_HANDx, F_HANDy)

        F_PROX_ARMx = -F_HANDx
        F_PROX_ARMy = -F_HANDy + (self.arm.mass * g)

        #self.arm.proximal_force = (F_PROX_ARMx, F_PROX_ARMy)

        F_SHOx = -F_PROX_ARMx
        F_SHOy = -F_PROX_ARMy

        #self.trunk.distal_force = (F_SHOx, F_SHOy)

        F_HIPx = -F_SHOx
        F_HIPy = -F_SHOy + (self.trunk.mass * g)

        #self.trunk.proximal_force = (F_HIPx, F_HIPy)

        F_PROX_THIGHx = -F_HIPx
        F_PROX_THIGHy = -F_HIPy

        #self.thigh.proximal_force = (F_PROX_THIGHx, F_PROX_THIGHy)

        F_KNEEx = -F_PROX_THIGHx
        F_KNEEy = -F_PROX_THIGHy + (self.thigh.mass * g)

        #self.thigh.distal_force = (F_KNEEx, F_KNEEy)

        F_PROX_SHANKx = -F_KNEEx
        F_PROX_SHANKy = -F_KNEEx

        #self.shank.proximal_force = (F_PROX_SHANKx, F_PROX_SHANKy)

        F_ANKx = -F_PROX_SHANKx
        F_ANKy = -F_PROX_SHANKy + (self.shank.mass * g)

        #self.shank.distal_force = (F_ANKx, F_ANKy)


        # Joint Moment Calculations
        L = self.arm.length
        R = L * self.arm.Dcom

        M_SHO = (self.arm.I * self.arm.alpha) - (F_PROX_ARMy * R * np.cos(np.radians(self.arm.angle))) \
            + (F_PROX_ARMx * R * np.sin(np.radians(self.arm.angle))) + (F_HANDy * (L - R) * np.cos(np.radians(self.arm.angle))) \
            - (F_HANDx * (L - R) * np.sin(np.radians(self.arm.angle)))

        M_DIST_TRUNK = -M_SHO

        L = self.trunk.length
        R = L * self.trunk.Dcom

        M_HIP = (self.trunk.I * self.trunk.alpha) + M_DIST_TRUNK - (F_SHOy * (L - R) * np.cos(np.radians(self.trunk.angle))) \
            + (F_SHOx * (L - R) * np.sin(np.radians(self.trunk.angle))) + (F_HIPy * R * np.cos(np.radians(self.trunk.angle))) \
            - (F_HIPx * R * np.sin(np.radians(self.trunk.angle)))

        M_PROX_THIGH = -M_HIP

        L = self.thigh.length
        R = L * self.thigh.Dcom

        M_KNEE = - (self.thigh.I * self.thigh.alpha) + M_PROX_THIGH - (F_PROX_THIGHy * R * -np.cos(np.radians(self.thigh.angle))) \
            - (F_PROX_THIGHx * R * np.sin(np.radians(self.thigh.angle))) + (F_KNEEy * (L - R) * -np.cos(np.radians(self.thigh.angle))) \
            + (F_KNEEx * (L - R) * np.sin(np.radians(self.thigh.angle)))

        M_PROX_SHANK = -M_KNEE

        L = self.shank.length
        R = L * self.shank.Dcom

        M_ANK = (self.shank.I * self.shank.alpha) - M_PROX_SHANK - (F_PROX_SHANKy * R * np.cos(np.radians(self.shank.angle))) \
            + (F_PROX_SHANKx * R * np.sin(np.radians(self.shank.angle))) + (F_ANKy * (L - R) * np.cos(np.radians(self.shank.angle))) \
            - (F_ANKx * (L - R) * np.sin(np.radians(self.shank.angle)))

        M_TOTAL = abs(M_SHO) + abs(M_HIP) + abs(M_KNEE) + abs(M_ANK)

        return [M_SHO, M_HIP, M_KNEE, M_ANK] , M_TOTAL

    def render_wireframe(self):

        heightFoot = self.height * .0425
        ankle = [0, heightFoot]
        knee = [ankle[0] + self.shank.length * self.shank.orientation[0,0],
                ankle[1] + self.shank.length * self.shank.orientation[0,1]]

        hip = [knee[0] + self.thigh.length * self.thigh.orientation[0,0],
               knee[1] + self.thigh.length * self.thigh.orientation[0,1]]

        shoulder = [hip[0] + self.trunk.length * self.trunk.orientation[0,0],
                    hip[1] + self.trunk.length * self.trunk.orientation[0,1]]

        hand = [shoulder[0] + self.arm.length * self.arm.orientation[0,0],
               shoulder[1] + self.arm.length * self.arm.orientation[0,1]]

        fig, ax = plt.subplots(figsize=(8,8))

        ax.plot([ankle[0],knee[0]],[ankle[1],knee[1]])
        ax.plot([knee[0],hip[0]],[knee[1],hip[1]])
        ax.plot([hip[0],shoulder[0]],[hip[1],shoulder[1]])
        ax.plot([shoulder[0],hand[0]],[shoulder[1],hand[1]])

        ax.legend(['Shank','Thigh','Trunk','Arm'])
        ax.set_xlim(-1,1)
        ax.set_ylim(0,2)
        ax.grid()

        plt.show()

    def create_joint(self, joint_radius=1, joint_center=np.array([0,0])):
        pts = 32
        xPts = np.zeros(pts)
        yPts = np.zeros(pts)

        for i in range(pts):
            ang = (i / (pts - 1)) * 2 * np.pi
            xPts[i] = np.cos(ang)
            yPts[i] = np.sin(ang)

        xPts *= joint_radius
        yPts *= joint_radius

        xPts += joint_center[0]
        yPts += joint_center[1]

        return xPts, yPts
            
    
    def render_outline(self, display_joints=False, display_bar=False, bar_pos=(0,0)):

        joint_radius = .025

        heightFoot = self.height * .0425
        ankle_pos = self.origin + np.array([0, heightFoot])

        shank_xPts, shank_yPts = self.shank.render_segment_outline(ankle_pos)

        knee_joint_center = ankle_pos + (self.shank.length + joint_radius) * np.dot(np.array([1,0]), self.shank.orientation)

        knee_xPts, knee_yPts = self.create_joint(joint_radius=joint_radius, joint_center=knee_joint_center)

        thigh_prox_point = knee_joint_center + joint_radius * np.dot(np.array([1,0]), self.thigh.orientation)

        thigh_xPts, thigh_yPts = self.thigh.render_segment_outline(thigh_prox_point)

        hip_joint_center = thigh_prox_point + (self.thigh.length + joint_radius) * np.dot(np.array([1,0]), self.thigh.orientation)

        hip_xPts, hip_yPts = self.create_joint(joint_radius=joint_radius, joint_center=hip_joint_center)

        trunk_prox_point = hip_joint_center + joint_radius * np.dot(np.array([1,0]), self.trunk.orientation)

        trunk_xPts, trunk_yPts = self.trunk.render_segment_outline(trunk_prox_point)

        shoulder_joint_center = trunk_prox_point + (self.trunk.length + joint_radius) * np.dot(np.array([1,0]), self.trunk.orientation)

        shoulder_xPts, shoulder_yPts = self.create_joint(joint_radius=joint_radius, joint_center=shoulder_joint_center)

        arm_prox_point = shoulder_joint_center + joint_radius * np.dot(np.array([1,0]), self.arm.orientation)

        arm_xPts, arm_yPts = self.arm.render_segment_outline(arm_prox_point)

        bar_xPts, bar_yPts = self.bar.render_segment_outline(location=bar_pos)

        fig, ax = plt.subplots(figsize=(8,8))

        ax.plot(shank_xPts, shank_yPts)
        ax.plot(thigh_xPts, thigh_yPts)
        ax.plot(trunk_xPts, trunk_yPts)
        ax.plot(arm_xPts, arm_yPts)

        if display_joints:
            ax.plot(knee_xPts,knee_yPts, color='k',alpha=.5)
            ax.plot(hip_xPts,hip_yPts, color='k',alpha=.5)
            ax.plot(shoulder_xPts,shoulder_yPts, color='k',alpha=.5)

        if display_bar:
            ax.plot(bar_xPts, bar_yPts, color='k')

        ax.set_xlim(-1,1)
        ax.set_ylim(0,2)
        ax.grid()

        plt.show()

