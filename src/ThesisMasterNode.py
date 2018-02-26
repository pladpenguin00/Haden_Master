#!/usr/bin/env python

import rospy
from std_msgs.msg import String

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import struct


from dynamic_reconfigure.server import Server as DynamicReconfigureServer

# Messages
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

# Configs
from Haden_Master.cfg import GoalConfig # this file would be cfg/GoalConfig.cfg
from pid_position_controller.cfg import Config


class ThesisMasterNode():
    def __init__(self):
        # __init__ is much like a c++ constructor function, implicitly called, and used to initialize things

        # Init our ROS Subscribers
        rospy.Subscriber("odom", Odometry, self.odom_callback)
        rospy.Subscriber("image", Image, self.image_callback)
        # Init our ROS Config server
        self.server = DynamicReconfigureServer(GoalConfig, self.config_callback)
        # Create a subscriber with appropriate topic, custom message and name of callback function.
        rospy.Subscriber(topic, node_example_data, cost_callback)
        rospy.Subscriber(topic, node_example_data, pos_callback)
        rospy.Subscriber(topic, node_example_data, curr_goal_callback)

        # By the time we get to the end of __init__, we might not even need a while loop. 
        #  If the only time we're "thinking" is when we get a new image, odometry, or goal, then we
        #  can just use those functions to run our program. Think "event-based" programming
        rospy.spin()
        
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
    def odom_callback(self, msg):
        # This contains code we'll execute when we get new odometry information
        # We'll get odometry info at a rate around 30Hz
        # We might just save this data as a class member to call when we get new images
        self.odom = msg

    def image_callback(self):
        # This contains code we'll execute when we receive a new image
        # This could be called up to 30Hz
        # This function will probably be the one that configs the pid_position_controller node
		
    def config_callback(self, config):
        # This contains code we'll execute when we receive a new goal from a user.
        # This will be called very rarely (only when the user provides new input)
		
    def listener():
		# Get the ~private namespace parameters from command line or launch file.
		topic = rospy.get_param('~topic', 'chatter')



    def rotate_translate(pos, goal, angle):
		#ox = pos.x
		#oy = pos.y
		gx = goal.x - pos.x
		gy = goal.y - pos.y 
	
	
	
		qx = 0 + math.cos(angle) * (gx - 0) - math.sin(angle) * (gy - 0)
		qy = 0 + math.sin(angle) * (gx - 0) - math.cos(angle) * (gy - 0)

		return {"x": qx, "y": qy}




    #MAIN PART O
    def output
    angle = rad * 180 / math.pi

    #where are we
    updated_goal = rotate_translate(origin, pos, angle)
		#output [x, y]
    #where to go
    heading = math.degrees(math.atan(updated_goal["y"]/updated_goal["x"]))

    #compare available heading based off cost
    path = np.arange(228, 140, -8)

    if updated_goal["y"] ** 2 + updated_goal["x"] ** 2 < 20:
        path = np.arange(195, 162, -3)

    mask = cost < 1250000
    available_path = path[mask]
    target_heading = min(available_path, key=lambda x:abs(x-heading)) - 180
        
	
    if all(mask)==False:
        nextpos= [ 0, 0, 180]
    if any(mask)==TRUE	
		if target_heading > 0
			a = 10
			A = abs(-90 + target_heading)
			X = abs(-180 + target_heading +90)
			nextx = math.sin(X) * a / math.sin(A) 
			nexty = math.sin(90) * a / math.sin(A)
		if target_heading < 0
			a = 10
			A = 90 + target_heading
			X = 180 + target_heading - 90
			nextx = math.sin(X) * a / math.sin(A) 
			nexty = math.sin(90) * a / math.sin(A)	
    
	
    pub.publish(nextx)
    pub.publish(nexty)	

if __name__ == '__main__':
    rospy.init_node("ThesisMasterNode")
    try:
        # This will initialize our class, calling ThesisMasterNode.__init__ implicitly
        tmn = ThesisMasterNode()
        listener()
        main()
    except rospy.ROSInterruptException:
        pass