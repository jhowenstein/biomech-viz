import math, time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


class Segment:

    # Segment inertial parameters for inverse dynamics
    Dcom = .5
    K = .5
    prop_mass = 1
    prop_length = 1

    # Segment diameter for outline rendering
    seg_radius = .06

    def __init__(self, length, prox_pt=(0,0), angle=0, mass=0):
        self.mass = mass
        self.length = length

        # initialize state parameters
        self.angle = angle
        self.prox_pt = prox_pt
        self.dist_pt = None
        self.alpha = 0

        # wire rendering parameters
        self.color = 'k'
        self.linewidth = 3

        self.children = []

    @property
    def I(self):
        return self.mass * self.K**2 * self.length**2

    @property
    def orientation(self):
        angle_rad = math.radians(self.angle)
        x1 = np.cos(angle_rad)
        x2 = np.sin(angle_rad)
        y1 = -np.sin(angle_rad)
        y2 = np.cos(angle_rad)
        return np.array([[x1, x2],[y1, y2]])

    def set_angle(self, new_angle):
        self.angle = new_angle

    def set_prox_pt(self, newProxPt):
        self.prox_pt = newProxPt

    def set_dist_pt(self, newDistPt):
        self.dist_pt = newDistPt

    def calc_distal_pt(self):
        return self.prox_pt + np.dot(np.array([self.length, 0]), self.orientation)

    def calc_proximal_pt(self):
        # Available for inverse kinematics
        return self.dist_pt - np.dot(np.array([self.length, 0]), self.orientation)

    def updateSegment(self, segOriginPt, new_angle, direction='prox'):
        if direction == 'prox':
            self.set_prox_pt(segOriginPt)
            self.set_angle(new_angle)
            self.dist_pt = self.calc_distal_pt()
        elif direction == 'dist':
            self.set_dist_pt(segOriginPt)
            self.set_angle(new_angle)
            self.prox_pt = self.calc_proximal_pt()

    def link_segment(self, segment):
        for child in self.children:
            if segment is child:
                print('Segment object already linked!')
                return
        self.children.append(segment)

    def unlink_segment(self, segment):
        for k, child in enumerate(self.children):
            if child is segment:
                self.children.pop(k)
                return
        print('Segments are not already linked!')

    def render_wireframe(self, ax, show_joints=False):
        # Currently only set up for proximal to distal calculation
        prox_pt = self.prox_pt
        dist_pt = self.calc_distal_pt()

        ax.plot([prox_pt[0],dist_pt[0]],
                [prox_pt[1],dist_pt[1]],
                 color=self.color, linewidth=self.linewidth)

        if show_joints:
            ax.scatter([dist_pt[0]],[dist_pt[1]],color=self.color)

        for child in self.children:
            child.set_prox_pt(dist_pt)
            child.render_wireframe(ax, show_joints=show_joints)

    def plot_wireframe(self, figsize=(6,6), axis_range=None, show=True, output=False, output_path='/figure.png',
                        title=None,show_joints=False, legend=None):

        if figsize is not None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots()

        self.render_wireframe(ax, show_joints=show_joints)
        ax.set_xlabel('X Position',fontsize=14)
        ax.set_ylabel('Y Position',fontsize=14)
        ax.grid()
        ax.set_title(title,fontsize=20)

        if axis_range is None:
            ax.axis('equal')
        else:
            ax.set_xlim(axis_range[0],axis_range[1])
            ax.set_ylim(axis_range[0],axis_range[1])

        if legend is not None:
            ax.legend(legend,fontsize=12)

        if output == True:
            plt.savefig(output_path)
        if show == True:
            plt.show()

    def render_segment_outline(self, seg_origin):
        """
        Returns the points to render the segment
        """
        xPts1 = []
        yPts1 = []
        xPts2 = []
        yPts2 = []

        x = np.array([1,0])
        y = np.array([0,1])

        curve_fractions = [0,.2,.5,.6,.6,.7,.8,.8,.9,.9]
        N = 100
        delta = 1/N
        for i in range(N):
            if i < 10:
                cross_radius = curve_fractions[i] * self.seg_radius
            elif i < 90:
                #cross_radius = self.length * .1
                cross_radius = self.seg_radius
            else:
                cross_radius = curve_fractions[99 - i] * self.seg_radius

            centerline = seg_origin + np.dot(x,self.orientation) * (i * delta * self.length)
            
            pt1 = centerline + cross_radius * np.dot(y, self.orientation)
            pt2 = centerline - cross_radius * np.dot(y, self.orientation)

            xPts1.append(pt1[0])
            yPts1.append(pt1[1])
            xPts2.append(pt2[0])
            yPts2.append(pt2[1])

        xPts2.reverse()
        yPts2.reverse()
        xPts = xPts1 + xPts2
        yPts = yPts1 + yPts2

        xPts = np.array(xPts)
        yPts = np.array(yPts)

        return xPts, yPts
        

class Segment_orig(object):  # Prev

    # Segment inertial parameters
    Dcom = .5
    K = .5
    prop_mass = 1
    prop_length = 1
    seg_radius = .06

    def __init__(self, length, angle, mass=0):
        self.mass = mass * self.prop_mass

        self.length = length * self.prop_length
        self.I = self.mass * self.K**2 * self.length**2

        self.alpha = 0


        self.angle = angle
        self.orientation = self.orientation_from_angle(angle)
        self.children = []

    def setAngle(self, new_angle):
        self.angle = new_angle
        self.orientation = self.orientation_from_angle(new_angle)

    def setProxPt(self, newProxPt):
        self.prox_pt = newProxPt

    def setDistPt(self, newDistPt):
        self.dist_pt = newDistPt

    def orientation_from_angle(self, theta):
        theta_rad = math.radians(theta)
        x1 = math.cos(theta_rad)
        x2 = math.sin(theta_rad)
        y1 = -math.sin(theta_rad)
        y2 = math.cos(theta_rad)
        return np.array([[x1, x2],[y1, y2]])

    def calc_distal_pt(self):
        distX = self.prox_pt[0] + (self.length * self.orientation[0,0])
        distY = self.prox_pt[1] - (self.length * self.orientation[0,1])
        return (distX, distY)

    def calc_proximal_pt(self):
        proxX = self.dist_pt[0] - (self.length * self.orientation[0,0])
        proxY = self.dist_pt[1] + (self.length * self.orientation[0,1])
        return (proxX, proxY)

    def updateSegment(self, segOriginPt, new_angle, direction='prox'):
        if direction == 'prox':
            self.setProxPt(segOriginPt)
            self.setAngle(new_angle)
            self.dist_pt = self.calc_distal_pt()
        elif direction == 'dist':
            self.setDistPt(segOriginPt)
            self.setAngle(new_angle)
            self.prox_pt = self.calc_proximal_pt()

    def link_segment(self, segment):
        for child in self.children:
            if segment is child:
                print('Segment object already linked!')
                return
        self.children.append(segment)

    def unlink_segment(self, segment):
        for k, child in enumerate(self.children):
            if child is segment:
                self.children.pop(k)
                return
        print('Segments are not already linked!')

    def render(self, ax):
        # Currently only set up for proximal to distal calculation
        prox_pt = self.prox_pt
        dist_pt = self.calc_distal_pt()

        ax.plot([prox_pt[0],dist_pt[0]],
                [prox_pt[1],dist_pt[1]],
                 color=self.color, linewidth=self.linewidth)

        for child in self.children:
            child.setProxPt(dist_pt)
            child.render(ax)

    def render_outline_pts(self, seg_origin):
        """
        Returns the points to render the segment
        """
        xPts1 = []
        yPts1 = []
        xPts2 = []
        yPts2 = []

        x = np.array([1,0])
        y = np.array([0,1])

        curve_fractions = [0,.2,.5,.6,.6,.7,.8,.8,.9,.9]
        N = 100
        delta = 1/N
        for i in range(N):
            if i < 10:
                cross_radius = curve_fractions[i] * self.seg_radius
            elif i < 90:
                #cross_radius = self.length * .1
                cross_radius = self.seg_radius
            else:
                cross_radius = curve_fractions[99 - i] * self.seg_radius

            centerline = seg_origin + np.dot(x,self.orientation) * (i * delta * self.length)
            
            pt1 = centerline + cross_radius * np.dot(y, self.orientation)
            pt2 = centerline - cross_radius * np.dot(y, self.orientation)

            xPts1.append(pt1[0])
            yPts1.append(pt1[1])
            xPts2.append(pt2[0])
            yPts2.append(pt2[1])

        xPts2.reverse()
        yPts2.reverse()
        xPts = xPts1 + xPts2
        yPts = yPts1 + yPts2

        xPts = np.array(xPts)
        yPts = np.array(yPts)

        return xPts, yPts

class Shank(Segment):

    Dcom = 1 - .433
    K = .302
    prop_mass = 0.0457
    prop_length = 0.252
    seg_radius = .06


class Thigh(Segment):

    Dcom = 1 - .433
    K = .323
    prop_mass = 0.1447
    prop_length = 0.2405
    seg_radius = .09


class Trunk(Segment):

    Dcom = .495
    K = .406
    prop_mass = 0.4302
    prop_length = 0.295
    seg_radius = .10


class Foot(Segment):
    pass


class Arm(Segment):

    Dcom = .530
    K = .368
    prop_mass = 0.04715
    prop_length = 0.35975
    seg_radius = .05


class Bar:
    def __init__(self, mass, accel=(0,0)):
        self.mass = mass
        self.x = 0
        self.y = 0
        self.accel_x = accel[0]
        self.accel_y = accel[1]

    def updateLocation(self, new_location):
        self.x = int(new_location[0])
        self.y = int(new_location[1])

    def render_outline_pts(self, location=(0,0)):
        bar_radius = .025
        pts = 32
        xPts = np.zeros(pts)
        yPts = np.zeros(pts)

        for i in range(pts):
            ang = (i / (pts - 1)) * 2 * np.pi
            xPts[i] = np.cos(ang)
            yPts[i] = np.sin(ang)

        xPts *= bar_radius
        yPts *= bar_radius

        xPts += location[0]
        yPts += location[1]

        return xPts, yPts

