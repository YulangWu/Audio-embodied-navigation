import logging
import os
from sys import platform

import yaml

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.render.profiler import Profiler
from igibson.utils.assets_utils import download_assets
import numpy as np

def main(selection="user", headless=False, short_exec=False):
    """
    Creates an iGibson environment from a config file with a turtlebot in Rs_int (interactive).
    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    # If they have not been downloaded before, download assets
    download_assets()
    config_filename = os.path.join(igibson.configs_path, "turtlebot_nav.yaml")
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    config_data["scene_id"] = "Rs_int"
    config_data["task"] = "point_nav_fixed" #"point_nav_random"
    # config_data["action_type"] = "discrete"
    # config_data["base"] = "JointController"
    print(config_data,'\n\n')
    # Reduce texture scale for Mac.
    if platform == "darwin":
        config_data["texture_scale"] = 0.5

    # Improving visuals in the example (optional)
    config_data["enable_shadow"] = True
    config_data["enable_pbr"] = True

    # config_data["load_object_categories"] = []  # Uncomment this line to accelerate loading with only the building
    env = iGibsonEnv(config_file=config_data, mode="gui_interactive" if not headless else "headless")
    max_iterations = 1 #10 if not short_exec else 1




    target_list = [[245,   311, 0], 
    [302,   360, 0],
    [300,   223, 0],
    [301,   404, 0],
    [244,   250, 0],
    [279,   195, 0],
    [252,   193, 0],
    [203,   295, 0],
    [268,   248, 0],
    [236,   366, 0],
    [169,   242, 0],
    [196,    98, 0],
    [319,   102, 0],
    [174,   283, 0],
    [133,   381, 0],
    [97,   298, 0],
    [297,   147, 0]]

    for i in range(len(target_list)):
        start_location = [0.64,1.22,0]
        target_location = target_list[i]
        print(target_location)
        env.reset()

        env.robots[0].set_position(start_location)
        env.task.target_pos = target_location 
        print("robot position = ",env.robots[0].get_position_orientation())
        print("target position = ",env.task.target_pos)

        #find the shortest path
        count = 0
        pre_pos = env.robots[0].get_position()
        final_path = []
        final_path.extend(pre_pos[:2]) #starting point
        while pre_pos[0] != env.task.target_pos[0] or pre_pos[1] != env.task.target_pos[1]:
            shortest_path = env.task.get_shortest_path(env)
            print("shortest path = ", shortest_path[0])
            for i in range(1,len(shortest_path[0])):
                final_path.extend(list(shortest_path[0][i]))


            pre_pos = []
            pre_pos.append(shortest_path[0][-1][0])
            pre_pos.append(shortest_path[0][-1][1])
            pre_pos.append(0)
            env.robots[0].set_position(pre_pos)
            count += 1
        
        print(list(final_path))
        np.savetxt(
            str(start_location[0]) + '_' + str(start_location[1]) + 
            'to' + 
            str(target_location[0]) + '_' + str(target_location[1]) + 
            'shortest_path.txt',final_path,delimiter=',')

    input('complete generating shortest path')



















    # for j in range(max_iterations):
    #     print("Resetting environment")
    #     env.reset()
    #     env.robots[0].set_position([-3.26,0.9,0])
    #     env.task.target_pos = [0.6600,   -1.4800,0] 
    #     print("robot position = ",env.robots[0].get_position_orientation())
    #     print("target position = ",env.task.target_pos)
    #     print("shortest path = ", env.task.get_shortest_path(env)[0])
    #     input('wait')
    #     # env.robots[0].action_type = "discrete"
    #     # env.robots[0].base = "position"
    #     # env.robots[0].control = "position"
    #     # env.robots[0]._default_controllers = "position"
    #     # env.robots[0]._default_base_joint_controller_config = "position"
    #     navigation_points = env.task.get_shortest_path(env)
        
    #     # print(dir(env.task))
    #     # print((dir(env.robots[0])))
    #     # print('1.',env.robots[0].controller_config)
    #     # print('2.',env.robots[0]._controllers)
    #     # print('3.',env.robots[0]._default_base_joint_controller_config)
    #     # print('4.',dir(env.robots[0].get_control_dict))
    #     # print('5.',env.robots[0].control_limits)

    #     # print('6.',dir(env.robots[0]._actions_to_control))
    #     # print('7.',env.robots[0]._default_base_differential_drive_controller_config)
    #     # print('8.',env.robots[0]._default_controllers)
    #     #env.robots[0].get_position_orientation() 
    #     old_pos = env.robots[0].get_position() 
    #     print(old_pos[0])
    #     input('action space')
    #     for i in range(1000):
    #         with Profiler("Environment action step"):
    #             # action = env.action_space.sample()
    #             dist = input('distance')
    #             ang = input('angle')
    #             action = [int(dist)/1000,int(ang)/1000]
      
    #             print('Action is  {}'.format(action))
                
    #             state, reward, done, info = env.step(action)
    #             new_pos = env.robots[0].get_position()
    #             print('old pos=',old_pos[0], old_pos[1])
    #             print('new pos=',new_pos[0], new_pos[1])
    #             print('new distance=',new_pos[0] - old_pos[0], new_pos[1] - old_pos[1])
    #             print('\n')
    #             old_pos = new_pos
                
    #             # if done:
    #             #     print("Episode finished after {} timesteps".format(i + 1))
    #             #     break
    env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
