import sys
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
gpu_idx = 0


class Logger: 
    def __init__(self, filename):
        self.terminal = sys.stdout 
        self.file = open(filename, "w")
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.file.write(message)
        self.file.flush()
    def flush(self):
        self.terminal.flush()
        self.file.flush()

def get_gpu_utilization():
    handle = nvmlDeviceGetHandleByIndex(gpu_idx)
    nvsmi = nvidia_smi.getInstance()
    return nvsmi.DeviceQuery()["gpu"][gpu_idx]



def run_stress_test(device="cuda"):

    x = torch.randn(2048, 2048).to(device)
    y = torch.randn(2048, 2048).to(device)

    last_time = math.floor(time.time())
    start_time = last_time
    
    print("time,matmul/s,utilization,graphics clock,temperature,power,fanspeed")
    with torch.no_grad():
        count = 0
        while True:
            torch.matmul(x, y)
            count += 1
            if math.floor(time.time()) > last_time:
                last_time = math.floor(time.time())
                timestamp = last_time - start_time
                stats = get_gpu_utilization() 

                util = stats["utilization"]["gpu_util"] 
                clocks = stats["clocks"]["graphics_clock"]
                temps = stats["temperature"]["gpu_temp"]
                power = stats["power_readings"]["power_draw"]
                fanspeed = stats["fan_speed"]

                print(f"{timestamp-1},{count},{util},{clocks},{temps},{power},{fanspeed}") 
                count = 0 

if __name__ == "__main__":

    nvsmi = nvidia_smi.getInstance() 
    pprint(nvsmi.DeviceQuery()["gpu"][gpu_idx])
    filename = nvsmi.DeviceQuery()["gpu"][gpu_idx]["product_name"] + ".csv"
    logger = Logger(filename)
    sys.stdout = logger 

    run_stress_test()
