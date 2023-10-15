import os
from sys import platform

def get_executable(executable_path):
    if platform == "linux" or platform == "linux2":
        ret = executable_path + 'CHANGE_THIS'
    elif platform == "darwin":
        ret = executable_path + 'exe_mac.app' 
    elif platform == "win32":
        ret = executable_path + 'exe_pc/UnityEnvironment.exe'

    return ret