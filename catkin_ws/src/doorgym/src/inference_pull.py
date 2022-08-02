#!/usr/bin/env python3

import argparse
import os
import sys
import numpy as np
import torch
import time
import rospy 
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler

sys.path.append("../DoorGym")
import a2c_ppo_acktr

from std_srvs.srv import Trigger, TriggerRequest
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from arm_operation.srv import * 
from arm_operation.msg import *
from ur5_bringup.srv import *
from gazebo_msgs.srv import *
from scipy.spatial.transform import Rotation as R

from curl_navi import DoorGym_gazebo_utils

class Inference:
    def __init__(self):

        self.method = rospy.get_param("~method")
        self.box = rospy.get_param("~box")

        self.joint_state_sub = rospy.Subscriber("/robot/joint_states", JointState, self.joint_state_cb, queue_size = 1)
        self.husky_vel_sub = rospy.Subscriber("/robot/cmd_vel", Twist, self.husky_vel_cb, queue_size=1)
        self.husky_cmd_pub = rospy.Publisher("/robot/cmd_vel", Twist, queue_size=1)
        self.get_knob_srv = rospy.ServiceProxy("/gazebo/get_link_state", GetLinkState)
        self.goto_joint_srv = rospy.ServiceProxy("/robot/ur5_control_server/ur_control/goto_joint_pose", joint_pose)
        self.goto_pose_srv = rospy.ServiceProxy("/robot/ur5_control_server/ur_control/goto_pose", target_pose)
        self.get_box_pos_srv = rospy.ServiceProxy("/gazebo/get_link_state", GetLinkState)
        self.arm_go_home = rospy.ServiceProxy("/robot/ur5/go_home", Trigger)
        self.gripper_close = rospy.ServiceProxy("/robot/gripper/close", Trigger)
        self.gripper_open = rospy.ServiceProxy("/robot/gripper/open", Trigger)
        self.get_pose_srv = rospy.ServiceProxy("/robot/ur5/get_pose", cur_pose)
        
        if(self.box):
            self.ran = rospy.ServiceProxy("/husky_ur5/pull_box", Trigger)
        else:
            self.ran = rospy.ServiceProxy("/husky_ur5/pull_cardboard", Trigger)
        self.joint = np.zeros(23)
        self.dis = 0
        self.listener = tf.TransformListener()
        self.joint_value = joint_value()

        if(self.method == "DoorGym"):
            model_path = DoorGym_gazebo_utils.download_model("1oNRt9NG6_KVVaLtRprA0LW-jvEgJYBSf", "../DoorGym", "ur5_pull")
        elif(self.method == "RL_mm"):
            model_path = DoorGym_gazebo_utils.download_model("1_7QLXH7s6VgwktPWLVc0k5gzLFVla70T", "../DoorGym", "husky_ur5_pull_3dof")
        elif(self.method == "6joints"):
            model_path = DoorGym_gazebo_utils.download_model("1NQMtSp7tF8qy6RqbBvfk4GqOfN1J-Vo0", "../DoorGym", "husky_ur5_pull")

        self.actor_critic = DoorGym_gazebo_utils.init_model(model_path, 23)
        
        self.actor_critic.to("cuda:0")
        self.recurrent_hidden_states = torch.zeros(1, self.actor_critic.recurrent_hidden_state_size)

        self.gripper_open()
        self.ran()
        self.arm_go_home()
        
        self.inference()

    def joint_state_cb(self, msg):

        self.joint_value.joint_value[0] = msg.position[7]
        self.joint_value.joint_value[1] = msg.position[6]
        self.joint_value.joint_value[2] = msg.position[5]
        self.joint_value.joint_value[3:] = msg.position[8:]

        self.joint[5:11] = self.joint_value.joint_value

        self.joint[15] = msg.velocity[7]
        self.joint[16] = msg.velocity[6]
        self.joint[17] = msg.velocity[5]
        self.joint[18:21] = msg.velocity[8:]

        self.joint[11:13] = msg.position[2]
        self.joint[21:23] = msg.velocity[2]
        
        trans = trans = [9.0, 14.18, 0.132]
        ros = [0.0, 0.0, -0.707, 0.707]

        try:
            trans, rot = self.listener.lookupTransform("/base_link", "/map", rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            print("Service call failed: %s"%e)

        _, _, yaw = euler_from_quaternion(rot)

        self.joint[3] = trans[0] - self.dis
        self.joint[4] = yaw
        
        self.dis = trans[0]

    def husky_vel_cb(self, msg):

        self.joint[13] = msg.linear.x
        self.joint[14] = msg.angular.z

    def get_distance(self):

        req = GetLinkStateRequest()
        req.link_name = "pull_box::knob"

        pos = self.get_knob_srv(req)

        trans = [9.11, 13.44, 0.859]

        try:
            trans, _ = self.listener.lookupTransform("/map", "/object_link", rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            print("Service call failed: %s"%e)

        self.joint[0] = trans[0] - pos.link_state.pose.position.x
        self.joint[1] = trans[1] - pos.link_state.pose.position.y
        self.joint[2] = trans[2] - pos.link_state.pose.position.z

    def go_to_knob(self):

        req = GetLinkStateRequest()
        req.link_name = "pull_box::knob"

        pos = self.get_knob_srv(req)

        trans = [9.11, 13.44, 0.859]
        quat = [0.502, -0.498, 0.502, -0.498]

        try:
            trans, quat = self.listener.lookupTransform("/map", "/object_link", rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            print("Service call failed: %s"%e)

        # calculate relationship between gripper and knob
        r = R.from_quat([quat])
        x = r.as_matrix()
        mat = np.identity(4)
        mat[0:3,0:3] = x
        mat[0:3,3] = trans
        
        r_knob = R.from_quat([pos.link_state.pose.orientation.x, pos.link_state.pose.orientation.y, pos.link_state.pose.orientation.z, pos.link_state.pose.orientation.w])
        x_knob = r_knob.as_matrix()
        mat_knob = np.identity(4)
        mat_knob[0:3,0:3] = x_knob
        mat_knob[0:3,3] = [pos.link_state.pose.position.x, pos.link_state.pose.position.y, pos.link_state.pose.position.z]

        ee_pose = np.dot(np.linalg.inv(mat), mat_knob)

        target_pose_req = target_poseRequest()
        target_pose_req.factor = 0.8

        res = self.get_pose_srv()
        target_pose_req.target_pose.position.x = res.pose.position.x + ee_pose[0,3] - 0.1
        target_pose_req.target_pose.position.y = res.pose.position.y + ee_pose[1,3] + 0.1
        target_pose_req.target_pose.position.z = res.pose.position.z + ee_pose[2,3] - 0.1
        target_pose_req.target_pose.orientation.x = res.pose.orientation.x
        target_pose_req.target_pose.orientation.y = res.pose.orientation.y
        target_pose_req.target_pose.orientation.z = res.pose.orientation.z
        target_pose_req.target_pose.orientation.w = res.pose.orientation.w
        self.goto_pose_srv(target_pose_req)

        target_pose_req.target_pose.position.x += 0.1
        self.goto_pose_srv(target_pose_req)

    # box
    # def distance(self, x1, y1, x2 = 9.0, y2 = 12.45):
    # cardboard
    def distance(self, x1, y1, x2 = 8.95, y2 = 12.10):

        return ((x1-x2)**2 + (y1-y2)**2)**0.5

    def inference(self):

        begin = time.time()
        closed = False

        while True: 

            target_pose_req = target_poseRequest()
            target_pose_req.factor = 0.8

            self.get_distance()
            joint_pose_req = joint_poseRequest()
            joint = torch.from_numpy(self.joint).float().to("cuda:0")
            action, self.recurrent_hidden_states = DoorGym_gazebo_utils.inference(self.actor_critic, joint, self.recurrent_hidden_states)
            next_action = action.cpu().numpy()[0,1,0]

            # husky ur5 pull parameter
            self.joint_value.joint_value[0] += next_action[2] * 0.002
            self.joint_value.joint_value[1] += next_action[3] * 0.001
            self.joint_value.joint_value[2] += next_action[4] * -0.001
            self.joint_value.joint_value[3] += next_action[5] * -0.001
            self.joint_value.joint_value[4] += next_action[6] * -0.001
            self.joint_value.joint_value[5] += next_action[7] * 0.001
            
            # husky
            t = Twist()

            # husky pull parameter
            
            t.linear.x = abs(next_action[0]) * 0.023
            t.angular.z = next_action[1] * 0.002

            res_box = self.get_box_pos_srv("pull_box::link_0","")

            if((self.joint[0] ** 2 + self.joint[1] **2) ** 0.5 <= 0.3 ):

                self.go_to_knob()
                self.gripper_close()
                rospy.sleep(1)
                res = self.get_pose_srv()
                if(not closed):
                    target_pose_req.target_pose.position.x = res.pose.position.x 
                    target_pose_req.target_pose.position.y = res.pose.position.y 
                    target_pose_req.target_pose.position.z = res.pose.position.z + 0.2
                    target_pose_req.target_pose.orientation.x = res.pose.orientation.x
                    target_pose_req.target_pose.orientation.y = res.pose.orientation.y
                    target_pose_req.target_pose.orientation.z = res.pose.orientation.z
                    target_pose_req.target_pose.orientation.w = res.pose.orientation.w
                    self.goto_pose_srv(target_pose_req)
                closed = True
            else:
                joint_pose_req.joints.append(self.joint_value)
                self.goto_joint_srv(joint_pose_req)
            
            if(closed):
                t.linear.x *= -1
                t.linear.z *= -1
                
            self.husky_cmd_pub.publish(t)
            
            if(self.distance(res_box.link_state.pose.position.x, res_box.link_state.pose.position.y) >= 2.0):
                self.gripper_open()
                self.arm_go_home()
                t.linear.z = -10.0
                self.husky_cmd_pub.publish(t)
                break

        end = time.time()

        print("time", end - begin)

if __name__ == '__main__':
    rospy.init_node("husky_ur5_pull_node", anonymous=False)
    inference = Inference()
    rospy.spin()