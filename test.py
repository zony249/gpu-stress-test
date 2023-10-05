import math
import time
from argparse import ArgumentParser
from pynvml import *
from pynvml.smi import nvidia_smi
from pprint import pprint
import psutil

import torch 
from torch import nn


nvmlInit()

def get_gpu_utilization():
    handle = nvmlDeviceGetHandleByIndex(0)
    nvsmi = nvidia_smi.getInstance()
    return nvsmi.DeviceQuery()["gpu"][0]



def run_stress_test(device="cuda"):

    layer = nn.Linear(1000, 1000, bias=False).to(device)
    x = torch.randn(1000, 1000).to(device)

    last_time = math.floor(time.time())
    start_time = last_time
    
    print("time,matmul/s,utilization,graphics clock,temperature,power")
    with torch.no_grad():
        count = 0
        while True:
            layer(x)
            count += 1
            line = ""
            if math.floor(time.time()) > last_time:
                last_time = math.floor(time.time())
                timestamp = last_time - start_time
                stats = get_gpu_utilization() 

                util = stats["utilization"]["gpu_util"] 
                clocks = stats["clocks"]["graphics_clock"]
                temps = stats["temperature"]["gpu_temp"]
                power = stats["power_readings"]["power_draw"]

                print(f"{timestamp-1},{count},{util},{clocks},{temps},{power}") 
                count = 0 

if __name__ == "__main__":

    run_stress_test()
