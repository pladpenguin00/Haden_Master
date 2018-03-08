#include "ros/ros.h"
#include "std_msgs/String.h"
#include <geometry_msgs/Twist.h>
#include <darknet_ros/bbox.h>
#include <darknet_ros/bbox_array.h>
#include <darknet_ros/objects.h>
#include <darknet_ros/persons.h>
#include <vector>

//std::string arlobot_number = "number";

darknet_ros::bbox_array box;
void objectCallback(const darknet_ros::bbox_array::ConstPtr& array)
{
	box = *array;

	int size_array = box.bboxes.size();
	//ROS_INFO("bbox array size %d", size_array);
	size_array = 0;
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "xlocation");
	ros::NodeHandle nh;

	ros::Subscriber yolo_sub = nh.subscribe("YOLO_bboxes", 1, objectCallback);
	ros::Publisher object = nh.advertise<darknet_ros::objects>("objects", 2);
	ros::Publisher people = nh.advertise<darknet_ros::persons>("people", 2);

	ros::Rate loop_rate(15);

	int xmax = 640;
	int xmid = xmax/2;
	int ymax = 460;
	int ymid = ymax/2;
	int hyst = 100;

	float xlocation;
	float ylocation;


	ROS_INFO("STARTING");

	darknet_ros::bbox stringA;
	stringA.Class = "person";; 
	stringA.prob = 0.60;
	stringA.xmin = 150;
	stringA.xmax = 350;
	stringA.ymin = 150;
	stringA.ymax = 300;

	darknet_ros::objects objects;
	darknet_ros::persons peoples;

	while (ros::ok())
	{
		int peopleCount = 0;
		int size = box.bboxes.size();
//		darknet_ros::bbox_array people;
//		darknet_ros::bbox_array object;
		ROS_INFO("Size of bbox array is %d", size);

		float peeps[10];
		int n = 0;
		float obj[10];
		int m = 0;

		for(int k = 0; k < 9; k++)
		{
			peeps[k] = 0;
			obj[k] = 0;
		}

		for(int i = 0; i < size; i++)
		{
			if(box.bboxes[i].Class == stringA.Class) //"person"
			{
				//peopleCount++;
				//people.bboxes.push_back(box.bboxes[i]);
				peeps[n] = ((box.bboxes[i].xmin + box.bboxes[i].xmax)/2);
				n++;
			}
			else
			{
				obj[m] = ((box.bboxes[i].xmin + box.bboxes[i].xmax)/2);
				m++;
			}
		}
		
		peoples.person1 = peeps[0];
		peoples.person2 = peeps[1];
		peoples.person3 = peeps[2];
		peoples.person4 = peeps[3];
		peoples.person5 = peeps[4];
		peoples.person6 = peeps[5];
		peoples.person7 = peeps[6];
		peoples.person8 = peeps[7];

		objects.object1 = obj[0];
		objects.object2 = obj[1];
		objects.object3 = obj[2];
		objects.object4 = obj[3];
		objects.object5 = obj[4];
		objects.object6 = obj[5];
		objects.object7 = obj[6];
		objects.object8 = obj[7];

		people.publish(peoples);
		object.publish(objects);
		
//		size = 0;
		
		ros::spinOnce();
		loop_rate.sleep();
	}

	return 0;
}
