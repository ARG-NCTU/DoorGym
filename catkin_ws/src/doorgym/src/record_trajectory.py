#!/usr/bin/env python3

import rospy
from gazebo_msgs.srv import *
from gazebo_msgs.msg import *
from std_msgs.msg import String
import yaml
import os

class record:
    def __init__(self):

        self.method = rospy.get_param("~method")
        self.pull = rospy.get_param("~pull", False)
        self.box = rospy.get_param("~box", False)

        self.timer = rospy.Timer(rospy.Duration(0.5), self.re_tra)
        self.fin_sub = rospy.Subscriber("/finish", String, self.fin_cb, queue_size = 1)
        self.get_robot_pos = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        self.total_traj = []
        self.tra = []
        self.my_dir = os.path.abspath(os.path.dirname(__file__))
        self.enable = False

    def fin_cb(self, msgs):

        if(msgs.data == "start"):
            self.enable = True

        if(msgs.data == "finished"):
            self.total_traj.append(self.tra)
            self.tra = []
            self.enable = False

        if(msgs.data == "end"):

            obj = ""

            if(self.pull):
                action = "_pull"
                if(self.box):
                    obj = "_box"
                else:
                    obj = "_cardboard"
            else:
                action = ""

            tra = {'environment' : "room_door", "policy": self.method, "trajectories" : self.total_traj}

            with open(os.path.join(self.my_dir,"../../../../Results/"+ self.method + obj + action + "_trajectory.yaml"), "w") as f:

                yaml.dump(tra, f)

    def re_tra(self, event):

        if(self.enable):
            robot_pose = self.get_robot_pos("robot", "")
            r_pose = {"position" : [robot_pose.pose.position.x, robot_pose.pose.position.y, robot_pose.pose.position.z],
                        "orientation" : [robot_pose.pose.orientation.x, robot_pose.pose.orientation.y, robot_pose.pose.orientation.z, robot_pose.pose.orientation.w]}
            self.tra.append(r_pose)

        

if __name__ == '__main__':
    rospy.init_node("record_trajectory_node", anonymous=False)
    record_trajectory = record()
    rospy.spin()