import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import (SideChannel, IncomingMessage, OutgoingMessage)
import numpy as np
import uuid

#------------------------------SideChannel---------------------------------
class StringLogChannel(SideChannel):

    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        print(msg.read_string())

    def send_string(sefl, data: str) -> None:
        msg = OutgoingMessage()
        msg.write_string(data)
        super().queue_message_to_send(msg)

#---------------------------------------------------------------------------

# #Used for testing/example use
# def main():
#     env = start_env(executable_file="C:/Users/oyo12/3D Objects/exe_filer/UnityEnvironment.exe")
#     """
#     joint 1, joint 1,... --> 10 steps 
#     joint 2, joint 2,..
#     ...
#     12 joints
#     """
#     individ = np.array([np.array([0.6714489 , 0.71748277, 0.76134361, 0.8025932 , 0.84081938, 0.8756402 , 0.90670775, 0.93371161, 0.95638197, 0.97449231]),
#                         np.array([0.6714489 , 0.71748277, 0.76134361, 0.8025932 , 0.84081938, 0.8756402 , 0.90670775, 0.93371161, 0.95638197, 0.97449231]),
#                         np.array([0.6714489 , 0.71748277, 0.76134361, 0.8025932 , 0.84081938, 0.8756402 , 0.90670775, 0.93371161, 0.95638197, 0.97449231]),
#                         np.array([0.6714489 , 0.71748277, 0.76134361, 0.8025932 , 0.84081938, 0.8756402 , 0.90670775, 0.93371161, 0.95638197, 0.97449231]),
#                         np.array([0.78232124, 0.82210884, 0.85867805, 0.89166345, 0.92073549, 0.94560368, 0.96601954, 0.98177909, 0.99272486, 0.99874749]),
#                         np.array([0.78232124, 0.82210884, 0.85867805, 0.89166345, 0.92073549, 0.94560368, 0.96601954, 0.98177909, 0.99272486, 0.99874749]),
#                         np.array([0.78232124, 0.82210884, 0.85867805, 0.89166345, 0.92073549, 0.94560368, 0.96601954, 0.98177909, 0.99272486, 0.99874749]),
#                         np.array([0.78232124, 0.82210884, 0.85867805, 0.89166345, 0.92073549, 0.94560368, 0.96601954, 0.98177909, 0.99272486, 0.99874749]),
#                         np.array([0.64116062, 0.66105442, 0.67933902, 0.69583173, 0.71036775, 0.72280184, 0.73300977, 0.74088955, 0.74636243, 0.74937375]),
#                         np.array([0.64116062, 0.66105442, 0.67933902, 0.69583173, 0.71036775, 0.72280184, 0.73300977, 0.74088955, 0.74636243, 0.74937375]),
#                         np.array([0.64116062, 0.66105442, 0.67933902, 0.69583173, 0.71036775, 0.72280184, 0.73300977, 0.74088955, 0.74636243, 0.74937375]),
#                         np.array([0.64116062, 0.66105442, 0.67933902, 0.69583173, 0.71036775, 0.72280184, 0.73300977, 0.74088955, 0.74636243, 0.74937375])])

#     individ2 = np.array([np.array([-50, -50, -50, -50 , -50, -50 , -50, -50, -50, -50]),
#                          np.array([-50, -50, -50, -50 , -50, -50 , -50, -50, -50, -50]),
#                          np.array([-50, -50, -50, -50 , -50, -50 , -50, -50, -50, -50]),
#                          np.array([-50, -50, -50, -50 , -50, -50 , -50, -50, -50, -50]),
#                          np.array([-50, -50, -50, -50, -50, -50, -50, -50, -50, -50]),
#                          np.array([-50, -50, -50, -50, -50, -50, -50, -50, -50, -50]),
#                          np.array([-50, -50, -50, -50, -50, -50, -50, -50, -50, -50]),
#                          np.array([-50, -50, -50, -50, -50, -50, -50, -50, -50, -50]),
#                          np.array([-50, -50, -50, -50, -50, -50, -50, -50, -50, -50]),
#                          np.array([-50, -50, -50, -50, -50, -50, -50, -50, -50, -50]),
#                          np.array([-50, -50, -50, -50, -50, -50, -50, -50, -50, -50]),
#                          np.array([-50, -50, -50, -50, -50, -50, -50, -50, -50, -50])])


#     all_actions = np.array([individ])

#     for _ in range(20):
#         send_actions_to_unity(env, all_actions)
#     env.close()
class UnityInterface():

    def __init__(self, executable_file: str = None, no_graphics: bool = True):
        self.env = self.start_env(executable_file=executable_file, no_graphics=no_graphics)


    def start_env(self, executable_file: str = None, no_graphics: bool = True) -> UnityEnvironment:
        """Starting a unity environment. 

        Args:
            executable_file (str, optional): Name of the executable file. Defaults to None, for runing in editor mode
        Returns:
            UnityEnvironment: return the unity environment
        """
        string_log = StringLogChannel()
        env = UnityEnvironment(file_name=executable_file, no_graphics=no_graphics, side_channels=[string_log])
        env.reset()
        return env

    def send_actions_to_unity(self, actions: np.array) -> list:

        self.env.reset()

        behavior_names = list(self.env.behavior_specs.keys())
        num_agents = len(behavior_names)
        num_actions = len(actions)
        positions = [] #[start pos, end pos]
        
        if num_actions != num_agents:
            print(f"Need more actions, training with {num_agents} agents!")
            return
        
        transposed_array = []
        for action in actions:
            transposed_array.append(np.transpose(action))
            
        for i in range(len(transposed_array[0])):
            for k in range(num_agents):
                decision_steps, _ = self.env.get_steps(behavior_names[k])
                agent_action = transposed_array[k]
                step = np.array([agent_action[i]])
                action_tuple = ActionTuple()
                action_tuple.add_continuous(step)
                self.env.set_actions(behavior_names[k], action_tuple)

                if i == 0:
                    positions.append([decision_steps.obs[0][:, :3]])
            self.env.step()

        for j in range(num_agents):
            decision_steps, _ = self.env.get_steps(behavior_names[j])
            positions[j].append(decision_steps.obs[0][:, :3])
         
        return positions

def stop_env(env) -> None:
    env.close()  

def reset_env(env) -> None:
    env.reset()




if __name__ == "__main__":
    main()
