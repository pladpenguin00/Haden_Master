#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String

import os
import sys
import random
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
import serial
import struct


from config import Config
import utils
import model as modellib
import visualize

from numpy.polynomial import Polynomial as P


class DepthCodeNode(object):
    def __init__(self):
		rospy.Subscriber("picture_timestamp", array, picture_timestamp_listener)
		rospy.Subscriber("pos", int32, pos_listener)



	def bbox_listener(data):
		rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

	def picture_timestamp_listener(data):
		rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

		
	def pos_listener(data):
		rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)


	def recognized_object_listener(data):
		rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)



	def distance_estimation()



		#Get Boundary box info and Odom info
		#Store the image


		newimage= 'Picture(%d)' % time
	
	
	

		imgdata[newimage] = {}
		imgdata[newimage]['time'] = time
		imgdata[newimage]['pos'] = pos
		
		#Multiple inputs from topics in array
		rospy.Subscriber("rocognized_object", string, recognized_object_listener)
		rospy.Subscriber("bbox_array", array, bbox_listener)
		
		countobj=len(bbox);
		for i in xrange(0,countobj):
			imgdata[newimage]['recognized_object'] = recognized	
			imgdata[newimage]['bbox'] = bbox
	
		if len(imgdata)=1
			OGimg=newimage;
			OGheight1=imgdata[newimage]['bbox'][0]
			OGheight2=imgdata[newimage]['bbox'][2]
			HsizeOG=OGheight2-OGheight1
			OGwidth1=imgdata[newimage]['bbox'][1]
			OGwidth2=imgdata[newimage]['bbox'][3]
			WsizeOG=OGwidth2-OGwidth1
	
	


		
		#Compare the image taken recently to the original image
		#Over time the image most recently taken should give the most accurate estimation
	
	
		#NO INFO Version
		i2height1=imgdata[newimage]['bbox'][0]
		i2height2=imgdata[newimage]['bbox'][2]
		Hsizei2=i2height2-i2height1
	
		i2width1=r2['rois'][i2search][1]
		i2width2=r2['rois'][i2search][3]
		Wsizei2=i2width2-i2width1

		Hdistance=distancemoved/(1-(HsizeOG/Hsizei2))
		Wdistance=distancemoved/(1-(WsizeOG/Wsizei2))
		avgdistance=(Hdistance+Wdistance)/2
		finalD=round(Hdistance,2)
		#finalD=round(avgdistance,2)
		stringD=str(finalD)

	
	
	
		#INFO VERSION
	
		#What is the object and how big is the average Method 2: KNOW THE HEIGTH OF OBJECT
	
		#Creating Matrix Predictor-----------------------------------------------------
		Drange=list(range(1, 100))
		DrangeView = np.array([item*FOV for item in Drange], dtype=np.float)

		xx = np.array([0] * 99 , dtype=np.float)
		yy = xx + 1
		zz = [item*sizeofobject for item in yy]
	
		onemeter=zz/DrangeView

		A=[[[] for i in range(99)] for i in range(89)]
		for i in range(89):
			for j in range(99):
				number=0 
				A[i][j]=number
		count=1
		in2c=0
		in2r=0
		while in2c < 89:
			i=98
			while i-in2r > 0 :        
				A[in2c][i]=(onemeter[i-count])+(onemeter[i])
				i-=1   
			in2c +=1
			count +=1
			in2r +=1
	

		x=A[round(distancemoved)][round(distancemoved)+1:99]
		y=np.array(list(range(round(distancemoved), (round(distancemoved)+len(A[round(distancemoved)][round(distancemoved)+1:99])))))
		p = P.fit(x, y, 12)
		poly_coeffs = np.polyfit(x, y, 12)

		def solve_for_y(poly_coeffs, EstimatedDistance):
			pc = poly_coeffs.copy()
			pc[-1] -= EstimatedDistance
			return np.roots(pc)
	
		#--------------------------------------------------------------------------------------------------------------------------------------
	
		#Calculations on estimated distance of Method 2
		v2perceptionH=(Hsizei2+Hsizei1)/1080
		v2perceptionW=(Wsizei2+Wsizei1)/1920
		v2EstimatedistanceW=(solve_for_y(poly_coeffs, v2perceptionW))*100
		v2EstimatedistanceH=(solve_for_y(poly_coeffs, v2perceptionH))*100
		v2avgdistance=(v2EstimatedistanceH[0]+v2EstimatedistanceW[0])/2
		v2finalD=p(v2perceptionH)
		#v2finalD=p(v2perceptionH)
		#v2finalD=v2avgdistance
		v2stringD=str(v2finalD)


	
	
		#Response Output of topic	
	
	
		pub = rospy.Publisher('estimated distance_method_1', int32 )
		pub.publish(finalD)
		pub = rospy.Publisher('estimated distance_method_2', int32 )
		pub.publish(v2finalD)
	
	
	
		#CombinedfinalD= (v2finalD + finalD) / 2
		#CombinedstringD= str(CombinedfinalD)
		#print ("Verson 1: The " + obj + " was initially " + stringD + " meters away! Verson 2: The " + obj + " was " + v2stringD + " meters away!")
		#print ("Average: " + CombinedstringD + " meters")
		#DATAFILEtxt.write("Verson 1: The " + obj + " was initially " + stringD + " meters away! Verson 2: The " + obj + " was " + v2stringD + " meters away! AVG = " + CombinedstringD + " meters \r\n")

		#FfinalD=finalD-distancemoved
		#FstringD=str(FfinalD)
		#Fv2finalD=v2finalD-distancemoved
		#Fv2stringD=str(Fv2finalD)
		#FCombinedfinalD= (Fv2finalD + FfinalD) / 2
		#FCombinedstringD= str(FCombinedfinalD)
		#print ("Verson 1: The " + obj + " was finally " + FstringD + " meters away! Verson 2: The " + obj + " was " + Fv2stringD + " meters away!")
		#print ("Average: " + FCombinedstringD + " meters")
		#DATAFILEtxt.write("Verson 1: The " + obj + " was Finally " + FstringD + " meters away! Verson 2: The " + obj + " was " + Fv2stringD + " meters away! AVG = " + FCombinedstringD + " meters \r\n")
		#published_distance = "Verson 1: The " + obj + " was Finally " + FstringD + " meters away! Verson 2: The " + obj + " was " + Fv2stringD + " meters away! AVG = " + FCombinedstringD + " meters \r\n")
	
		#time1= time.time();
		#totaltime= time1-time0;
		#print(totaltime)
		#DATAFILEtxt.write("total time " + str(totaltime) + "\r\n")

		#error = abs((CombinedfinalD / ActualDistance) - 1) 
		#errorRound=round(error,2)
		#errorstringD=str(errorRound*CombinedfinalD)
		#print ("Error: " + errorstringD + " meters")

		#closePred= CombinedfinalD - CombinedfinalD*errorRound
		#farPred= CombinedfinalD + CombinedfinalD*errorRound	
		#farstringD=str(farPred)
		#closestringD=str(closePred)
	
	

	
	