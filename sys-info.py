import os
import platform
import psutil
import torch
import argparse

def summary():   

    o = f'''System Information:
-------------------
OS: {platform.system()} ({platform.release()})
Python version: {platform.python_version()}
PyTorch version: {torch.__version__}
CUDA toolkit version: {torch.version.cuda}
CUDNN version: {torch.backends.cudnn.version()}
CUDA availability: {torch.cuda.is_available()}
'''

    o += os.linesep
    
    total_vram = 0
    total_cuda_cores = 0
    for i in range(torch.cuda.device_count()):
        total_cuda_cores += torch.cuda.get_device_properties(i).multi_processor_count
        total_vram += torch.cuda.get_device_properties(i).total_memory
        
    o += f'''Total Resources:
------------------------
Logical CPUs: {psutil.cpu_count()}
RAM: {psutil.virtual_memory().total / 1024**3:.2f} GB
CUDA devices: {torch.cuda.device_count()}
CUDA cores: {total_cuda_cores}
GPU memory: {total_vram / 1024**3:.2f} GB'''
    return o

def cpu_info():
    return f'''CPU Information:
-------------------
Physical CPUs: {psutil.cpu_count(logical=False)}    
Logical CPUs: {psutil.cpu_count()}
CPU architecture: {platform.machine()}
CPU bits: {platform.architecture()[0]}'''
    
def cpu_usage():
    return f'''CPU Usage:
-------------------
CPU % usage: {psutil.cpu_percent()}%
CPU % usage per thread: {psutil.cpu_percent(percpu=True)}
RAM usage (used / total): {psutil.virtual_memory().used / 1024**3:.2f} GB / {psutil.virtual_memory().total / 1024**3:.2f} GB
RAM % usage: {psutil.virtual_memory().percent}%'''

def gpu_info():
    o = f'''GPU Information:
-------------------'''
    for i in range(torch.cuda.device_count()):
        o += f'''{os.linesep * (i > 0)}
GPU {i}: {torch.cuda.get_device_name(i)}
       Cuda cores: {torch.cuda.get_device_properties(i).multi_processor_count}
       Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB
       Compute capability: {torch.cuda.get_device_capability(i)}'''
    return o

def gpu_usage():
    o = f'''GPU Usage:
-------------------'''
    for i in range(torch.cuda.device_count()):
        # total_mem = torch.cuda.get_device_properties(i).total_memory
        # reserved_mem = torch.cuda.memory_reserved(i)
        (free_mem, total_mem) = torch.cuda.mem_get_info(i)
        used_mem = total_mem - free_mem
        o += f'''{os.linesep * (i > 0)}
GPU {i}: {torch.cuda.get_device_name(i)}
       Memory usage (used / total): {used_mem / 1024**3:.2f} GB / {total_mem / 1024**3:.2f} GB
       Memory % usage: {used_mem / total_mem * 100:.1f}%'''
    return o

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', action='store_true', help='Print summary')
    parser.add_argument('--cpu', action='store_true', help='Print CPU info')
    parser.add_argument('--cpu-usage', action='store_true', help='Print CPU usage')
    parser.add_argument('--gpu', action='store_true', help='Print GPU info')
    parser.add_argument('--gpu-usage', action='store_true', help='Print GPU usage')
    opt = parser.parse_args()

    if not any(vars(opt).values()):
        opt.summary = True

    args_map = {
        'summary': summary,
        'cpu': cpu_info,
        'cpu_usage': cpu_usage,
        'gpu': gpu_info,
        'gpu_usage': gpu_usage
    }

    output = []

    for key, value in args_map.items():
        if getattr(opt, key):
            output.append(value())

    for idx, item in enumerate(output):
        print(f'{os.linesep * (idx > 0)}{item}')