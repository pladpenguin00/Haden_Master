#include <ros/ros.h>
#include <std_msgs/String.h> 
#include <stdio.h>
#include <math.h>
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Vector3Stamped.h"
#include "std_msgs/Float64.h"

#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/CommandTOL.h>
#include <sensor_msgs/NavSatFix.h>
#include <mavros_msgs/Thrust.h>
#include <mavros_msgs/AttitudeTarget.h>

#include <tf2/LinearMath/Quaternion.h>

#include <darknet_ros/bbox.h>
#include <darknet_ros/bbox_array.h>

darknet_ros::bbox_array box;
void object_cb(const darknet_ros::bbox_array::ConstPtr& array)
{
	box = *array;

	// int size_array = box.bboxes.size();
	// ROS_INFO("bbox array size %d", size_array);
}

mavros_msgs::State current_state;
void state_cb(const mavros_msgs::State::ConstPtr& msg){
	current_state = *msg;
}

sensor_msgs::NavSatFix gps_reading;
void gps_cb(const sensor_msgs::NavSatFix::ConstPtr& msg){
	gps_reading = *msg;
}

int main(int argc, char **argv)
{
        ros::init(argc, argv, "attitude_one");
	ros::NodeHandle n;

	// bbox subscribe
	ros::Subscriber yolo_sub = n.subscribe("YOLO_bboxes", 1, object_cb);

	ros::Publisher pub_thr = n.advertise<mavros_msgs::Thrust>("/mavros/setpoint_attitude/thrust", 100);
	ros::Publisher att_pub  = n.advertise<mavros_msgs::AttitudeTarget>("/mavros/setpoint_raw/attitude", 100);
	ros::Subscriber state_sub = n.subscribe<mavros_msgs::State>("mavros/state", 10, state_cb);
	ros::ServiceClient arming_client = n.serviceClient<mavros_msgs::CommandBool>("mavros/cmd/arming");
	ros::Subscriber gps_sub = n.subscribe<sensor_msgs::NavSatFix>("mavros/global_position/global", 30, gps_cb);

	ros::Rate loop_rate(100);

//	mavros_msgs::Thrust cmd_thr;
	
	mavros_msgs::AttitudeTarget cmd_att;
	cmd_att.thrust = 0.5;

	cmd_att.type_mask = 
			mavros_msgs::AttitudeTarget::IGNORE_ROLL_RATE | //;
			mavros_msgs::AttitudeTarget::IGNORE_PITCH_RATE;
	//		mavros_msgs::AttitudeTarget::IGNORE_YAW_RATE; // |
	//		mavros_msgs::AttitudeTarget::IGNORE_THRUST; 
	//		mavros_msgs::AttitudeTarget::IGNORE_ATTITUDE; // thrust!

	mavros_msgs::CommandBool arm_cmd;
	arm_cmd.request.value = true;
	//arming_client.call(arm_cmd);

	// Image frame
	int xmax = 640;
	int xmid = xmax/2;
	int ymax = 460;
	int ymid = ymax/2;
	int hyst = 100;

	int xlocation;
	int ylocation;

	darknet_ros::bbox string;
	string.Class = "person";
	string.prob = 0.60;
	string.xmin = 150; string.xmax = 350;
	string.ymin = 150; string.ymax = 300;


	float rate = 0.0;
	
	while(ros::ok()){
		int peopleCount = 0;
		int size = box.bboxes.size();
		darknet_ros::bbox_array people;
//		ROS_INFO("Size of bbox array is %d", size);

		xlocation = 0;

		for(int i = 0; i < size; i++)
		{
			if(box.bboxes[i].Class == string.Class)
			{
				peopleCount++;
				people.bboxes.push_back(box.bboxes[i]);
			}
		}

//		ROS_INFO("Number of people is %d", peopleCount);
		
		if(peopleCount == 0)
		{
			rate = 0.3;
		}
		else if(peopleCount == 1)
		{
			// follow
			xlocation = (people.bboxes[0].xmin + people.bboxes[0].xmax) / 2;
			ROS_INFO("xlocation = %d", xlocation);
			
			if(xlocation > (xmid + hyst))
			{
				rate = 0.3;
			}
			else if(xlocation < (xmid - hyst))
			{	
				rate = -0.3;
			}
			else
			{
				rate = 0.0;
			}
		}
		else
		{
			rate = 0.0;
		}
		
		//Create attitude command message
		//cmd_att.header.stamp = ros::Time::now();
		//cmd_att.header.seq=count;
		//cmd_att.header.frame_id = 1;
		//cmd_att.body_rate.x = 0.1;//0.001*some_object.position_x;
		//cmd_att.body_rate.y = 0.3;//0.001*some_object.position_y;
		cmd_att.body_rate.z = rate;//0.001*some_object.position_z;

		tf2::Quaternion quat;
		quat.setRPY(0.0, 0.0, 10.0);

		cmd_att.orientation.x = quat.x();
		cmd_att.orientation.y = quat.y();
		cmd_att.orientation.z = quat.z();
		cmd_att.orientation.w = quat.w();
		att_pub.publish(cmd_att);

		people.bboxes.clear();

		ros::spinOnce();
		loop_rate.sleep();
	}
	return 0;
}


