import os
import platform
import psutil
import GPUtil
import torch
import argparse
from tabulate import tabulate

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class CpuInfo:
    def __init__(self):
        self.cpus_physical = psutil.cpu_count(logical=False)
        self.cpus_logical = psutil.cpu_count()
        self.cpu_architecture = platform.machine()
        self.cpu_bits = platform.architecture()[0]
        self.ram_total = round(psutil.virtual_memory().total / 1024**3, 2)

    def load(self):
        return round(psutil.cpu_percent(), 2)

    def load_per_thread(self):
        return psutil.cpu_percent(percpu=True)

    def ram_used(self):
        return round(self.ram_total - psutil.virtual_memory().available / 1024**3, 2)

    def ram_free(self):
        return round(psutil.virtual_memory().available / 1024**3, 2)

    def ram_utilization(self):
        return round(psutil.virtual_memory().percent, 2)


class Gpu:
    def __init__(self, id):
        self.id = id
        self.cuda_cores = torch.cuda.get_device_properties(
            id).multi_processor_count
        self._nvidia_smi = GPUtil.getGPUs()[id]
        if self._nvidia_smi:
            self.name = self._nvidia_smi.name
            self.mem_total = round(self._nvidia_smi.memoryTotal / 1024, 2)
            self.driver_version = self._nvidia_smi.driver
        else:
            self.name = torch.cuda.get_device_name(id)
            self.mem_total = round(
                torch.cuda.mem_get_info(self.id)[1] / 1024**3, 2)
            self.driver_version = 'N/A'

    def load(self):
        return round(self._nvidia_smi.load * 100, 2) if self._nvidia_smi else 0

    def mem_used(self):
        if self._nvidia_smi:
            return round(self._nvidia_smi.memoryUsed / 1024, 2)
        else:
            (free_mem, total_mem) = torch.cuda.mem_get_info(self.id)
            return round((total_mem - free_mem) / 1024**3, 2)

    def mem_free(self):
        return round(self._nvidia_smi.memoryFree / 1024 if self._nvidia_smi else torch.cuda.mem_get_info(self.id)[0] / 1024**3, 2)

    def mem_utilization(self):
        return round(self._nvidia_smi.memoryUtil * 100 if self._nvidia_smi else (self.mem_used() / self.mem_total) * 100, 2)


class GpuInfo:
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.cuda_devices = torch.cuda.device_count()

        nvidia_smi_gpus = GPUtil.getGPUs()
        if len(nvidia_smi_gpus) > 0:
            self.gpus = [Gpu(gpu.id) for gpu in nvidia_smi_gpus]
        else:
            self.gpus = [Gpu(i) for i in range(self.cuda_devices)]

        self.cuda_cores_total = sum([gpu.cuda_cores for gpu in self.gpus])
        self.mem_total = sum([gpu.mem_total for gpu in self.gpus])

    def load(self):
        return sum([gpu.load() for gpu in self.gpus]) / len(self.gpus)

    def load_per_gpu(self):
        return [gpu.load() for gpu in self.gpus]

    def mem_used(self):
        return sum([gpu.mem_used() for gpu in self.gpus])

    def mem_used_per_gpu(self):
        return [gpu.mem_used() for gpu in self.gpus]

    def mem_free(self):
        return sum([gpu.mem_free() for gpu in self.gpus])

    def mem_free_per_gpu(self):
        return [gpu.mem_free() for gpu in self.gpus]

    def mem_utilization(self):
        return round(self.mem_used() / self.mem_total * 100, 2)


class SysInfo:
    def __init__(self):
        self.os = platform.system()
        self.python_version = platform.python_version()
        self.pytorch_version = torch.__version__
        self.cuda_version = torch.version.cuda
        self.cudnn_version = torch.backends.cudnn.version()

        self._cpu_info = CpuInfo()
        self._gpu_info = GpuInfo()
        self._header_map = {
            'cpu_info': ['Physical CPUs', 'Logical CPUs', 'CPU Architecture', 'CPU Bits', 'RAM Total (GB)'],
            'gpu_info': ['ID', 'Name', 'CUDA Cores', 'Memory Total (GB)', 'Driver Version'],
            'gpu_summary': ['CUDA Devices', 'CUDA Cores', 'Memory Total (GB)'],
            'cpu_usage': ['Load (%)', 'RAM Utilization (%)', 'RAM Used (GB)', 'RAM Free (GB)'],
            'gpu_usage': ['ID', 'Name', 'Load (%)', 'Memory Utilization (%)', 'Memory Used (GB)', 'Memory Free (GB)'],
            'gpu_usage_summary': ['Load (%)', 'Memory Utilization (%)', 'Memory Used (GB)', 'Memory Free (GB)'],
        }

    def sys_summary(self, header=False):
        data = [
            ['OS', self.os],
            ['PyTorch', self.pytorch_version],
            ['Python', self.python_version],
            ['CUDA', self.cuda_version],
            ['cuDNN', self.cudnn_version],
            ['CUDA Available', self._gpu_info.cuda_available],
            ['Logical CPUs', self._cpu_info.cpus_logical],
            ['RAM Total (GB)', self._cpu_info.ram_total],
            ['CUDA Devices', self._gpu_info.cuda_devices],
            ['CUDA Cores', self._gpu_info.cuda_cores_total],
            ['GPU Memory Total (GB)', self._gpu_info.mem_total],
        ]
        return data

    def cpu_info(self, header=False):
        data = [self._cpu_info.cpus_physical, self._cpu_info.cpus_logical,
                self._cpu_info.cpu_architecture, self._cpu_info.cpu_bits, self._cpu_info.ram_total]

        if header:
            return [self._header_map['cpu_info'], data]
        return [data]

    def gpu_info(self, header=False):
        data = [[gpu.id, gpu.name, gpu.cuda_cores, gpu.mem_total,
                 gpu.driver_version] for gpu in self._gpu_info.gpus]

        if header:
            return [self._header_map['gpu_info']] + data
        return [data]

    def gpu_info_summary(self, header=False):
        data = [self._gpu_info.cuda_devices,
                self._gpu_info.cuda_cores_total, self._gpu_info.mem_total]

        if header:
            return [self._header_map['gpu_summary'], data]
        return [data]

    def cpu_usage(self, header=False):
        data = [self._cpu_info.load(), self._cpu_info.ram_utilization(),
                self._cpu_info.ram_used(), self._cpu_info.ram_free()]

        if header:
            return [self._header_map['cpu_usage'], data]
        return [data]

    def gpu_usage(self, header=False):
        data = [[gpu.id, gpu.name, gpu.load(), gpu.mem_utilization(
        ), gpu.mem_used(), gpu.mem_free()] for gpu in self._gpu_info.gpus]

        if header:
            return [self._header_map['gpu_usage']] + data
        return [data]

    def gpu_usage_summary(self, header=False):
        data = [self._gpu_info.load(), self._gpu_info.mem_utilization(),
                self._gpu_info.mem_used(), self._gpu_info.mem_free()]

        if header:
            return [self._header_map['gpu_usage_summary'], data]
        return [data]


if __name__ == '__main__':
    sysinfo = SysInfo()

    parser = argparse.ArgumentParser()
    parser.add_argument('--sys-summary', action='store_true',
                        help='Print system summary')
    parser.add_argument('--cpu-info', action='store_true',
                        help='Print CPU info')
    parser.add_argument('--cpu-usage', action='store_true',
                        help='Print CPU usage')
    parser.add_argument('--gpu-info', action='store_true',
                        help='Print GPU info')
    parser.add_argument('--gpu-usage', action='store_true',
                        help='Print GPU usage')
    parser.add_argument('--gpu-usage-summary', action='store_true',
                        help='Print GPU usage summary')
    opt = parser.parse_args()

    if not any(vars(opt).values()):
        opt.sys_summary = True

    args_map = {
        'sys_summary': sysinfo.sys_summary,
        'cpu_info': sysinfo.cpu_info,
        'cpu_usage': sysinfo.cpu_usage,
        'gpu_info': sysinfo.gpu_info,
        'gpu_usage': sysinfo.gpu_usage,
        'gpu_usage_summary': sysinfo.gpu_usage_summary,
    }

    for key, value in args_map.items():
        if getattr(opt, key):
            data = value(header=True)
            print(tabulate(data, headers='firstrow' if key != 'sys_summary' else '', tablefmt='grid',
                  numalign='center', stralign='center' if key != 'sys_summary' else 'left'))
