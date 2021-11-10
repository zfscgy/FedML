from typing import List, Callable
from multiprocessing import Process, Manager


from FedML.Base.config import GlobalConfig


def parallel_process(funcs: List[Callable], devices: List[str] = None):
    manager = Manager()
    return_dict = manager.dict()
    if devices is None:
        devices = ['cuda' for _ in funcs]

    def with_device(i: int, return_dict: dict):
        GlobalConfig.device = devices[i]
        print(f"Current process: {i}, device: {GlobalConfig.device}")
        return_dict[i] = funcs[i]()

    current_processes = []

    for i, _ in enumerate(funcs):
        current_processes.append(Process(target=with_device, args=(i, return_dict)))
        current_processes[-1].start()

    for p in current_processes:
        p.join()

    return [return_dict[i] for i in range(len(funcs))]
