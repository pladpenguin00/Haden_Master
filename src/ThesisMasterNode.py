#!/usr/bin/env python

import rospy
from std_msgs.msg import String

import os
import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import serial
import struct




curr_pos =



origin = (0,0)


def cost_callback(data):
    # Simply print out values in our custom message.
	global cost
	cost = data
    

def pos_callback(data):
    # Simply print out values in our custom message.
	global cost
	cost = data

def curr_goal_callback(data):
    # Simply print out values in our custom message.
	global cost
	cost = data

def listener():
    # Get the ~private namespace parameters from command line or launch file.
    topic = rospy.get_param('~topic', 'chatter')


    # Create a subscriber with appropriate topic, custom message and name of callback function.
    rospy.Subscriber(topic, node_example_data, cost_callback)
    rospy.Subscriber(topic, node_example_data, pos_callback)
    rospy.Subscriber(topic, node_example_data, curr_goal_callback)

    # Wait for messages on topic, go to callback function when new messages arrive.
    rospy.spin()

def rotate_translate(pos, goal, angle):
    ox, oy = pos
    gx, gy = goal

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) - math.cos(angle) * (py - oy)

    return qx, qy


while True:

    listener()

    angle = rad * 180 / math.pi

    #where are we
    updated_goal = rotate_translate(origin, pos, angle)
		#output [x, y]
    #where to go
    heading = math.degrees(math.atan(updated_goal[1]/updated_goal[0]))

    #compare available heading based off cost
    path = np.arange(228, 140, -8)

    if updated_goal[1] ** 2 + updated_goal[0] ** 2 < 20:
        path = np.arange(195, 162, -3)

    mask = cost < 1250000
    available_path = path[mask]
    target_heading = min(available_path, key=lambda x:abs(x-heading)) - 180
        #left is negative right is positive

    if not any(mask)==False:
        target_heading = 180

    pub = rospy.Publisher('target_heading', int64, queue_size=1)
    pub.publish(target_heading)
