import multiprocessing as mp
import random
import sys
import os

from omegaconf import OmegaConf

sys.path.append("../../")

from src.distributed import Launcher


def main():
    model_configs = [
        "../../configs/model/hclt_64.yaml",
        "../../configs/model/pd_64.yaml",
        "../../configs/model/pdhclt_2_64.yaml",
        "../../configs/model/pdhclt_2_128.yaml",
        "../../configs/model/pdhclt_4_64.yaml",
        "../../configs/model/pdhclt_4_128.yaml",
        "../../configs/model/hclt_128.yaml",
        "../../configs/model/pd_128.yaml",
        "../../configs/model/hclt_256.yaml"
    ]
    data_configs = [
        "../../configs/data/celeba.yaml", 
        "../../configs/data/imagenet.yaml"
        # "../../configs/data/imagenet32.yaml"
    ]
    optim_configs = [
        "../../configs/optim/annealed_strategy.yaml",
        # "../../configs/optim/fixed_lr1.yaml",
        "../../configs/optim/fixed_lr2.yaml",
        # "../../configs/optim/full_batch_only.yaml"
    ]

    tasks = []
    for model_config in model_configs:
        for data_config in data_configs:
            for optim_config in optim_configs:
                task = ("run_cmd", (f"python main.py --data-config {data_config} --model-config {model_config} --optim-config {optim_config}",))
                tasks.append(task)

    random.shuffle(tasks)

    launcher = Launcher(device_ids = [0,1,2,3,4,5])

    launcher.run_tasks(tasks)


if __name__ == "__main__":
    mp.set_start_method("forkserver")
    main()