#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

# FROM THE MONODEPTH CODE-------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import matplotlib.pyplot as plt
import cv2
import time

from monodepth_model import *
from monodepth_dataloader import *
from average_gradients import *
from ThesisMasterNode import *


parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

parser.add_argument('--encoder',          type=str,   help='type of encoder, vgg or resnet50', default='vgg')
parser.add_argument('--image_path',       type=str,   help='path to the image', default="/home/nvidia/monodepth-master/")
parser.add_argument('--checkpoint_path',  type=str,   help='path to a specific checkpoint to load', default="/home/nvidia/monodepth-master/")
parser.add_argument('--input_height',     type=int,   help='input height', default=256)
parser.add_argument('--input_width',      type=int,   help='input width', default=512)

args = parser.parse_args()

#--------------------------------------------------------
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


class ThesisMasterNode(object):
    def __init__(self):
        # __init__ is much like a c++ constructor function, implicitly called, and used to initialize things

        # Init our ROS Subscribers
        rospy.Subscriber("odom", Odometry, self.odom_callback)
        
        # Init our ROS Config server
        self.server = DynamicReconfigureServer(GoalConfig, self.config_callback)
        # Create a subscriber with appropriate topic, custom message and name of callback function.
        rospy.Subscriber("odom_pos", pos, self.pos_callback)
        rospy.Subscriber("goal", curr_goal, self.curr_goal_callback)
        rospy.Subscriber("image", Image, self.image_callback)
        pubx = rospy.Publisher("x_pos", int64, queue_size=1)
        puby = rospy.Publisher("y_pos", int64, queue_size=1)
        # By the time we get to the end of __init__, we might not even need a while loop. 
        #  If the only time we're "thinking" is when we get a new image, odometry, or goal, then we
        #  can just use those functions to run our program. Think "event-based" programming
        rospy.spin()
        
    def post_process_disparity(disp):
		_, h, w = disp.shape
		l_disp = disp[0,:,:]
		r_disp = np.fliplr(disp[1,:,:])
		m_disp = 0.5 * (l_disp + r_disp)
		l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
		l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
		r_mask = np.fliplr(l_mask)
		return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


    def test_simple(params):
		"""Test function."""

		left  = tf.placeholder(tf.float32, [2, args.input_height, args.input_width, 3])
		model = MonodepthModel(params, "test", left, None)
    
		#t0=time.time()
		input_image = scipy.misc.imread(args.image_path, mode="RGB")
		original_height, original_width, num_channels = input_image.shape
		width=512
		height=256
		input_image=cv2.resize(input_image,(width,height))
		#input_image = scipy.misc.imresize(input_image, [args.input_height, args.input_width], interp='lanczos')
		input_image = input_image.astype(np.float32) / 255
		input_images = np.stack((input_image, np.fliplr(input_image)), 0)
		#thisimage=tf.Variable(input_images, name='thisimage')

		# SESSION
		config = tf.ConfigProto(allow_soft_placement=True)
		sess = tf.Session(config=config)

		# SAVER
		train_saver = tf.train.Saver()

		# INIT
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		coordinator = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

		# RESTORE
		restore_path = args.checkpoint_path.split(".")[0]
		train_saver.restore(sess, restore_path)

		
    

    def pos_callback(data):
		# Simply print out values in our custom message.
		global pos
		pos = data

    def curr_goal_callback(data):
		# Simply print out values in our custom message.
		global curr_goal
		curr_goal = data
    def odom_callback(self, msg):
        # This contains code we'll execute when we get new odometry information
        # We'll get odometry info at a rate around 30Hz
        # We might just save this data as a class member to call when we get new images
        self.odom = msg

    def image_callback(self):
        # This contains code we'll execute when we receive a new image
        # This could be called up to 30Hz
        # This function will probably be the one that configs the pid_position_controller node
		image = self		
		t0=time.time()
		input_image = image
		original_height, original_width, num_channels = input_image.shape
		width=512
		height=256
		input_image=cv2.resize(input_image,(width,height))
		#input_image = scipy.misc.imresize(input_image, [args.input_height, args.input_width], interp='lanczos')
		input_image = input_image.astype(np.float32) / 255
		input_images = np.stack((input_image, np.fliplr(input_image)), 0)
		t1=time.time()
		total=t1-t0 
		#print(total)




		disp = sess.run(model.disp_left_est[0], feed_dict={left: input_images})
		disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)

		t1=time.time()
		total=t1-t0 
		#print(total)

		output_directory = os.path.dirname(args.image_path)
		output_name = os.path.splitext(os.path.basename(args.image_path))[0]

		#np.save(os.path.join(output_directory, "{}_disp.npy".format(output_name)), disp_pp)
		disp_to_img = scipy.misc.imresize(disp_pp.squeeze(), [256, 512])

		imgnum = np.matrix(disp_to_img)
		imgnum = np.array(imgnum.sum(axis=0))
		cost = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
		cost[0] = np.sum(imgnum[0][4:46])
		cost[1] = np.sum(imgnum[0][46:91])
		cost[2] = np.sum(imgnum[0][91:136])
		cost[3] = np.sum(imgnum[0][136:181])
		cost[4] = np.sum(imgnum[0][181:226])
		cost[5] = np.sum(imgnum[0][226:271])
		cost[6] = np.sum(imgnum[0][271:316])
		cost[7] = np.sum(imgnum[0][316:361])
		cost[8] = np.sum(imgnum[0][361:406])
		cost[9] = np.sum(imgnum[0][406:451])
		cost[10] = np.sum(imgnum[0][451:496])
		return cost
		
		
    def config_callback(self, config):
        # This contains code we'll execute when we receive a new goal from a user.
        # This will be called very rarely (only when the user provides new input)
		



    def rotate_translate(pos, goal, angle):
		#ox = pos.x
		#oy = pos.y
		gx = goal.x - pos.x
		gy = goal.y - pos.y 
	
	
	
		qx = 0 + math.cos(angle) * (gx - 0) - math.sin(angle) * (gy - 0)
		qy = 0 + math.sin(angle) * (gx - 0) - math.cos(angle) * (gy - 0)

		return {"x": qx, "y": qy}

    def mainparam(_):

		params = monodepth_parameters(
			encoder='vgg',
			height=256,
			width=512,
			batch_size=2,
			num_threads=1,
			num_epochs=1,
			do_stereo=False,
			wrap_mode="border",
			use_deconv=False,
			alpha_image_loss=0,
			disp_gradient_loss_weight=0,
			lr_loss_weight=0,
			full_summary=False)



		test_simple(params)

    
	
	
    def NewTarget(cost)
		#MAIN PART O

		angle = rad * 180 / math.pi

		#where are we
		updated_goal = rotate_translate(pos, curr_goal, angle)
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
			nextx = 0
			nexty = 0
			# if the nextx and nexty are 0 the robot needs to rotate 180 
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
        pubx.publish(nextx)
        puby.publish(nexty)	  
	


if __name__ == '__main__':
    rospy.init_node("ThesisMasterNode")
    try:
        # This will initialize our class, calling ThesisMasterNode.__init__ implicitly
        tmn = ThesisMasterNode()
        mainparam()
        NewTarget(cost)

    except rospy.ROSInterruptException:
        pass