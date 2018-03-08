#include "ros/ros.h"
#include "std_msgs/String.h"
#include <arlobot/motor_cmd.h>
#include <arlobot/ping.h>
#include <geometry_msgs/Twist.h>
#include <arlobot/bbox.h>
#include <arlobot/bbox_array.h>
#include <darknet_ros/bbox.h>
#include <darknet_ros/bbox_array.h>
#include <vector>

//std::string arlobot_number = "number";

arlobot::ping pings;
void pingCallback(const arlobot::ping::ConstPtr& msg)
{
	pings = *msg;
}

geometry_msgs::Twist keys;
void keyCallback(const geometry_msgs::Twist::ConstPtr& key)
{
	keys = *key;
}

geometry_msgs::Twist man_bot;
int manual;
void manualCallback(const geometry_msgs::Twist::ConstPtr& man)
{
	man_bot = *man;
	manual = man_bot.linear.x;
}


darknet_ros::bbox_array box;
void objectCallback(const darknet_ros::bbox_array::ConstPtr& array)
{
	box = *array;

	int size_array = box.bboxes.size();
	ROS_INFO("bbox array size %d", size_array);
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "arlobot");
	ros::NodeHandle nh;

	std::string motor = "motor";
	std::string ping = "ping";
	//std::string check = argv;
	ros::Publisher cmd_one = nh.advertise<arlobot::motor_cmd>("motors/one", 5);

	ros::Publisher cmd_two = nh.advertise<arlobot::motor_cmd>("motors/one", 5);
	ros::Subscriber ping_sub = nh.subscribe("ping/four", 5, pingCallback);
//	ros::Subscriber key_sub = nh.subscribe("key", 1, keyCallback);
//	ros::Subscriber manual_sub = nh.subscribe("manual", 1, manualCallback);
	ros::Subscriber yolo_sub = nh.subscribe("YOLO_bboxes", 1, objectCallback);

	ros::Rate loop_rate(5);

	arlobot::motor_cmd commands;	
	int lvl1 = 30;
	int lvl2 = 20;
	int lvl3 = 10;

//	float linear = 0.0;
//	float angular = 0.0;

	// Image frame
	int xmax = 640;
	int xmid = xmax/2;
	int ymax = 460;
	int ymid = ymax/2;
	int hyst = 100;

	int xlocation;
	int ylocation;

	pings.f3;

	ROS_INFO("STARTING");

	darknet_ros::bbox stringA;
	stringA.Class = "person";; 
	stringA.prob = 0.60;
	stringA.xmin = 150;
	stringA.xmax = 350;
	stringA.ymin = 150;
	stringA.ymax = 300;

	while (ros::ok())
	{
		int peopleCount = 0;
		int size = box.bboxes.size();
		darknet_ros::bbox_array people;
		ROS_INFO("Size of bbox array is %d", size);

		for(int i = 0; i < size; i++)
		{
			if(box.bboxes[i].Class == stringA.Class) //"person"
			{
				peopleCount++;
				people.bboxes.push_back(box.bboxes[i]);
			}
		}

		ROS_INFO("Number of people is %d", peopleCount);

		if(peopleCount == 0)
		{
			// keep rotating until you find someone
			commands.left = 6000; //700;
			commands.right = 6000; //500;
			cmd_one.publish(commands);
			ROS_INFO("Looking");
		}
/*		else if(peopleCount == 2)
		{
			// stop
			commands.left = 6000;
			commands.right = 6000;
			cmd_one.publish(commands);
		}*/
		else if(peopleCount == 1)
		{
			// follow
			xlocation = (people.bboxes[0].xmin + people.bboxes[0].xmax) / 2;
			if(xlocation > (xmid + hyst)) // person to right
			{
				commands.left = 6500;
				commands.right = 5500;
				cmd_one.publish(commands);
				ROS_INFO("Turning Right");
			}
			else if(xlocation < (xmid - hyst)) // person to left
			{
				commands.left = 5500;
				commands.right = 6500;
				cmd_one.publish(commands);
				ROS_INFO("Turning Left");
			}
			else // go stop
			{
				commands.left = 6000;
				commands.right = 6000;
				cmd_one.publish(commands);
				ROS_INFO("Found Person!");
			}
			
		}
		else
		{
			//stop
			commands.left = 6000;
			commands.right = 6000;
			cmd_one.publish(commands);
			ROS_INFO("Stoping, too many people");
		}

/*
		switch(manual)
		{
			case(6):
				cmd_one.publish(commands);
				break;
			case(7):
				cmd_two.publish(commands);
				break;
			case(8):
				//cmd_pub.publish(cmd_att);
				break;
			default:
				commands.left = 6000;
				commands.right = 6000;
				cmd_one.publish(commands);
				cmd_two.publish(commands);
				break;
		}
	
*/
		people.bboxes.clear(); // Clear array

		ros::spinOnce();
		loop_rate.sleep();
	}

	return 0;
}
