#!/usr/bin/env python

import rospy
from std_msgs.msg import String

import sys
import math
import struct
import numpy as np

class MaintainDistanceNode(object):
    def __init__(self):
	    
#Initial
	#collect 100 bbox heights 

	count=0
	while count < 100
		rospy.Subscriber("bbox", bbox, self.bbox_callback)
		
		height = bbox_bottom - bbox_top
		np.insert(a, count, height)
		count=count+1
		#find distribution

		
		#find standard deviation 
		std_dev100 = stdev(a)
		mu100=mean(a)
		plus100=mu100+std_dev100
		minus100=mu100-std_dev100
		
	rospy.Subscriber("bbox", bbox, self.bbox_callback)
	rospy.spin()
	
#LOOP

 def bbox_callback(data):
		# Simply print out values in our custom message.
		global height
		height = data
		
		main(code)
		
 def check_height()
	#collect 10 bbox heights
	count=0	
	b=0
	while count < 10
		rospy.Subscriber("bbox", bbox, self.bbox_callback)
	
		height = bbox_bottom - bbox_top
		np.insert(b, count, height)
		count=count+1
		#find distribution

		#find standard deviation
		std_dev10 = stdev(a)
		mu10=mean(a)
		plus10=mu10+std_dev10
		minus10=mu10+std_dev10

		
 def maincode()	 #if the height gets bigger back up till back at height
	check_height()
	while plus10 > plus100
		motor=-0.1
		check_height()
		#if the height gets smaller move forward till back at height
	while minus10 < minus100 
		motor=0.1
		check_height()
	#keep object in center of image
	#	if bbox moves to higher x then rotate to the right till back at center
#if bbox moves to lower x then rotate to the left till back at center



if __name__ == '__main__':
    rospy.init_node("MaintainDistance")
    try:
        # This will initialize our class, calling ThesisMasterNode.__init__ implicitly
        tmn = MaintainDistanceNode()


	except rospy.ROSInterruptException:
	   pass