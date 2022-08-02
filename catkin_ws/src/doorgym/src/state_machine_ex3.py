#!/usr/bin/env python3

import rospy
import smach
import smach_ros
import time
import os
import sys
import numpy as np
import torch
import rospy 
import tf
import random
import tensorflow
import math
from tf.transformations import euler_from_quaternion, quaternion_from_euler

sys.path.append("../DoorGym")
import a2c_ppo_acktr

from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger, TriggerRequest
from sensor_msgs.msg import JointState, LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Bool, String, Float32
from arm_operation.srv import * 
from arm_operation.msg import *
from ur5_bringup.srv import *
from gazebo_msgs.srv import *
from gazebo_msgs.msg import *

from curl_navi import DoorGym_gazebo_utils
import yaml

pub_info = rospy.Publisher('/state', String, queue_size=10)
finish_info = rospy.Publisher('/finish', String, queue_size=10)
goal_pub = rospy.Publisher("/tare/goal", PoseStamped, queue_size=1)

my_dir = os.path.abspath(os.path.dirname(__file__))
# read yaml
with open(os.path.join(my_dir,"../../../../Data/goal.yaml"), 'r') as f:
    data = yaml.load(f)

goal_totoal = data['goal']
goal = []

enable = False

# metric
count = 0
total = len(goal_totoal)
success = 0
coi = 0

begin = 0

class loop(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['loop_done', 'looping'])

    def execute(self, userdata):

        global count, total, success, coi

        if(count == total):
            # finish all goal

            # calculate metric
            s_r = (success/total) * 100
            f_r = 100 - s_r
            a_c = coi / total

            # output result 
            d = {'success_rate':s_r, "fail_rate":f_r, "average_coillision":a_c}

            with open(os.path.join(my_dir,"../../../../Data/" + method + "_pull_result.yaml"), "w") as f:
                yaml.dump(d, f)

            finish_info.publish("end")
            
            rospy.loginfo('End')
            return 'loop_done'
        else:
            goal = goal_totoal[count]
            return 'looping'

class init(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['init_done'])
        self.arm_home_srv = rospy.ServiceProxy("/robot/ur5/go_home", Trigger)
        self.set_init_pose_srv = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self.gripper_open = rospy.ServiceProxy("/robot/gripper/open", Trigger)

    def execute(self, userdata):

        global begin

        begin = time.time()

        rospy.loginfo("init position")

        # req = SetModelStateRequest()
        req = ModelState()
        req.model_name = 'robot'
        req.pose.position.x = 9.0 
        req.pose.position.y = 14.2
        req.pose.position.z = 0.1323
        req.pose.orientation.x = 0.0
        req.pose.orientation.y = 0.0
        req.pose.orientation.z = -0.707
        req.pose.orientation.w = 0.707

        # set robot
        self.set_init_pose_srv(req)

        if(box):
            # set box position
            req = ModelState()
            req.model_name = 'pull_box'
            req.pose.position.x = 8.96242507935
            req.pose.position.y = 12.4507133386
            req.pose.position.z = 0.57499964254
            req.pose.orientation.x = 1.57141918775e-07
            req.pose.orientation.y = -1.8141635704e-07
            req.pose.orientation.z = 0.000529657457698
            req.pose.orientation.w = 1.0

            # set box
            self.set_init_pose_srv(req)
        else:
            # set cardboard
            req = ModelState()
            req.model_name = 'pull_box'
            req.pose.position.x = 8.95191862859
            req.pose.position.y = 12.1063873859
            req.pose.position.z = 0.575066995938
            req.pose.orientation.x = -0.0013749844125
            req.pose.orientation.y = 3.13900522696e-07
            req.pose.orientation.z = -1.68650047537e-05
            req.pose.orientation.w = 1.0

            # set cardboard
            self.set_init_pose_srv(req)
        
        # set gripper
        self.gripper_open()

        # arm go home
        self.arm_home_srv()

        return 'init_done'


class pull(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['pulling'])

        self.joint_state_sub = rospy.Subscriber("/robot/joint_states", JointState, self.joint_state_cb, queue_size = 1)
        self.husky_vel_sub = rospy.Subscriber("/robot/cmd_vel", Twist, self.husky_vel_cb, queue_size=1)
        self.husky_cmd_pub = rospy.Publisher("/robot/cmd_vel", Twist, queue_size=1)
        self.get_knob_srv = rospy.ServiceProxy("/gazebo/get_link_state", GetLinkState)
        self.goto_joint_srv = rospy.ServiceProxy("/robot/ur5_control_server/ur_control/goto_joint_pose", joint_pose)
        self.goto_pose_srv = rospy.ServiceProxy("/robot/ur5_control_server/ur_control/goto_pose", target_pose)
        self.gripper_close = rospy.ServiceProxy("/robot/gripper/close", Trigger)
        self.gripper_open = rospy.ServiceProxy("/robot/gripper/open", Trigger)
        self.get_pose_srv = rospy.ServiceProxy("/robot/ur5/get_pose", cur_pose)
        self.arm_go_home = rospy.ServiceProxy("/robot/ur5/go_home", Trigger)
        self.sub_collision = rospy.Subscriber("/robot/bumper_states", ContactsState, self.cb_collision, queue_size=1)
        self.joint = np.zeros(23)
        self.dis = 0
        self.listener = tf.TransformListener()
        self.joint_value = joint_value()
        self.collision_states = False

        if(method == "DoorGym"):
            model_path = DoorGym_gazebo_utils.download_model("1oNRt9NG6_KVVaLtRprA0LW-jvEgJYBSf", "../DoorGym", "ur5_pull")
        elif(method == "RL_mm"):
            model_path = DoorGym_gazebo_utils.download_model("1_7QLXH7s6VgwktPWLVc0k5gzLFVla70T", "../DoorGym", "husky_ur5_pull_3dof")
        elif(method == "6joints"):
            model_path = DoorGym_gazebo_utils.download_model("1NQMtSp7tF8qy6RqbBvfk4GqOfN1J-Vo0", "../DoorGym", "husky_ur5_pull")

        self.actor_critic = DoorGym_gazebo_utils.init_model(model_path, 23)
        
        self.actor_critic.to("cuda:0")
        self.recurrent_hidden_states = torch.zeros(1, self.actor_critic.recurrent_hidden_state_size)

    def execute(self, userdata):

        global enable

        finish_info.publish("start")

        if(method == "RL_mm"):

            target_pose_req = target_poseRequest()
            target_pose_req.factor = 0.8

            res = self.get_pose_srv()

            self.get_distance()
            joint_pose_req = joint_poseRequest()
            joint = torch.from_numpy(self.joint).float().to("cuda:0")
            action, self.recurrent_hidden_states = DoorGym_gazebo_utils.inference(self.actor_critic, joint, self.recurrent_hidden_states)
            next_action = action.cpu().numpy()[0,1,0]

            target_pose_req.target_pose.position.x = res.pose.position.x + 0.001 * next_action[2]
            target_pose_req.target_pose.position.y = res.pose.position.y - 0.0002 * next_action[3]
            target_pose_req.target_pose.position.z = res.pose.position.z + 0.0007 * next_action[4]
            target_pose_req.target_pose.orientation.x = res.pose.orientation.x
            target_pose_req.target_pose.orientation.y = res.pose.orientation.y
            target_pose_req.target_pose.orientation.z = res.pose.orientation.z
            target_pose_req.target_pose.orientation.w = res.pose.orientation.w
   
            # husky
            t = Twist()

            # husky push parameter
            
            t.linear.x = abs(next_action[0]) * 0.023
            t.angular.z = next_action[1] * 0.003

            if((self.joint[0] ** 2 + self.joint[1] **2) ** 0.5 <= 0.4):

                if(not enable):
                    self.go_to_knob()
                t.linear.x *= -1
                t.linear.z *= -1
                enable = True
            else:
                self.goto_pose_srv(target_pose_req)
                self.gripper_open()
                if(enable):
                    self.arm_go_home()
                enable = False
                
            self.husky_cmd_pub.publish(t)

        else:

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

            if((self.joint[0] ** 2 + self.joint[1] **2) ** 0.5 <= 0.4):

                if(not enable):
                    self.go_to_knob()
                t.linear.x *= -1
                t.linear.z *= -1
                enable = True
            else:
                joint_pose_req.joints.append(self.joint_value)
                self.goto_joint_srv(joint_pose_req)
                self.gripper_open()
                if(enable):
                    self.arm_go_home()
                enable = False
                
            self.husky_cmd_pub.publish(t)

        return 'pulling'

    def cb_collision(self, msg):

        global coi
        count = 0
        if self.collision_states == True:
            if msg.states == [] and count > 1000:
                self.collision_states = False
            else:
                count += 1
        elif msg.states != [] and count == 0:
            self.collision_states = True
            coi += 1
        else:
            self.collision_states = False
            count = 0

    def get_odom(self, msg):

        trans = [0.0]

        try:
            trans, rot = self.listener.lookupTransform("/base_link", "/map", rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            print("Service call failed: %s"%e)

        _, _, yaw = euler_from_quaternion(rot)

        ori = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]

        _, _, odom_yaw = euler_from_quaternion(ori)

        self.joint[3] = trans[0] - msg.pose.pose.position.x - self.dis
        self.joint[4] = yaw - odom_yaw

        self.dis = self.joint[3]  

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
        target_pose_req.target_pose.position.y = res.pose.position.y + ee_pose[1,3] + 0.12
        target_pose_req.target_pose.position.z = res.pose.position.z + ee_pose[2,3] - 0.10
        target_pose_req.target_pose.orientation.x = res.pose.orientation.x
        target_pose_req.target_pose.orientation.y = res.pose.orientation.y
        target_pose_req.target_pose.orientation.z = res.pose.orientation.z
        target_pose_req.target_pose.orientation.w = res.pose.orientation.w
        self.goto_pose_srv(target_pose_req)

        target_pose_req.target_pose.position.x += 0.1
        self.goto_pose_srv(target_pose_req)

        self.gripper_close()
        rospy.sleep(1)

        target_pose_req.target_pose.position.z += 0.1
        self.goto_pose_srv(target_pose_req)

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

class is_pull(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['not_yet', 'pulled'])
        self.gripper_open = rospy.ServiceProxy("/robot/gripper/open", Trigger)
        self.arm_go_home = rospy.ServiceProxy("/robot/ur5/go_home", Trigger)
        self.pub_cmd = rospy.Publisher("/robot/cmd_vel", Twist, queue_size=1)
        self.get_box_pos_srv = rospy.ServiceProxy("/gazebo/get_link_state", GetLinkState)
        self.husky_cmd_pub = rospy.Publisher("/robot/cmd_vel", Twist, queue_size=1)

    def distance(self, x1, y1):
        if(box):
            x2, y2 = 9.0, 12.45
        else:
            x2, y2 = 8.95, 12.10

        return ((x1-x2)**2 + (y1-y2)**2)**0.5

    def execute(self, userdata):

        global begin

        # husky
        t = Twist()

        res_box = self.get_box_pos_srv("pull_box::link_0","")

        if(self.distance(res_box.link_state.pose.position.x, res_box.link_state.pose.position.y) >= 2.0):
            self.gripper_open()
            t.linear.z = -10.0
            self.husky_cmd_pub.publish(t)
            self.arm_go_home()
            return 'pulled'
        elif(time.time() - begin >= 180):
            count += 1
            return 'pulled'
        else:
            return 'not_yet'

class Navigation(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['navigating'])

    def execute(self, userdata):

        global goal_totoal, count

        # publish goal to tare
        pose = PoseStamped()

        pose.header.frame_id = "map"
        pose.pose.position.x = goal_totoal[count][0]
        pose.pose.position.y = goal_totoal[count][1]

        goal_pub.publish(pose)

        pub_info.publish("nav")
        return 'navigating'

class is_goal(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['not_yet', 'navigated'])
        
        self.get_robot_pos = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
    
    def execute(self, userdata):

        global success, begin
        global goal_totoal, count

        self.goal = np.array([goal_totoal[count][0], goal_totoal[count][1]]) 
        robot_pose = self.get_robot_pos("robot","")

        x, y = robot_pose.pose.position.x, robot_pose.pose.position.y
        dis = np.linalg.norm(self.goal - np.array([x, y]))

        if(dis < 0.8):
            pub_info.publish("stop")
            finish_info.publish("finished")
            success += 1
            count += 1
            return 'navigated'
        elif(time.time() - begin >= 180):
            finish_info.publish("finished")
            count += 1
            return 'navigated'
        else:
            return 'not_yet'

def main():

    rospy.init_node("experiment3_node", anonymous=False)

    sm = smach.StateMachine(outcomes=['end'])

    global method, box

    method = rospy.get_param("~method")
    box = rospy.get_param("~box")

    with sm:
        smach.StateMachine.add('loop', loop(), transitions={'looping':'init', 'loop_done':'end'})
        smach.StateMachine.add('init', init(), transitions={'init_done':'pull'})
        smach.StateMachine.add('pull', pull(), transitions={'pulling':'is_pull'})
        smach.StateMachine.add('is_pull', is_pull(), transitions={'pulled':'nav_to_goal', 'not_yet':'pull'})
        smach.StateMachine.add('nav_to_goal', Navigation(), transitions={'navigating':'is_goal'})
        smach.StateMachine.add('is_goal', is_goal(), transitions={'not_yet':'nav_to_goal', 'navigated':'loop'})

    # Create and start the introspection server
    sis = smach_ros.IntrospectionServer('my_smach_introspection_server', sm, '/SM_ROOT')
    sis.start()
    
    # Execute SMACH plan
    outcome = sm.execute()
    
    # Wait for ctrl-c to stop the application
    rospy.spin()
    sis.stop()

if __name__ == '__main__':
    main()