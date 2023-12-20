import os
import yaml
import random
import argparse
import setproctitle

import torch
import numpy as np

from runner import RUNNER

def get_args():
    parser = argparse.ArgumentParser(description="DRL")

    parser.add_argument("--env", type=str, default="mujoco")
    # other choices requires additive config file
    parser.add_argument("--env-name", type=str, default="Hopper-v3")

    # algorithm parameters
    parser.add_argument("--algo", type=str, default="diaster_ac")
    parser.add_argument("--hidden-dims", type=list, default=[256, 256])                 # dimensions of actor/critic hidden layers
    parser.add_argument("--actor-lr", type=float, default=3e-4)                         # learning rate of actor
    parser.add_argument("--critic-lr", type=float, default=3e-4)                        # learning rate of critic
    parser.add_argument("--epired-lr", type=float, default=3e-4)                        # learning rate of episodic return decomposition
    parser.add_argument("--gamma", type=float, default=0.99)                            # discount factor
    parser.add_argument("--tau", type=float, default=0.005)                             # update rate of target network
    # (for sac)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)                         # learning rate of alpha

    # return decomposition parameters
    parser.add_argument("--epired", action="store_true", default=True)
    parser.add_argument("--epired-interval", type=int, default=int(1e3))
    parser.add_argument("--epired-batch-size", type=int, default=64)
    parser.add_argument("--return-scale", type=int, default=10)
    parser.add_argument("--return-shift", type=int, default=0)

    # replay-buffer parameters
    parser.add_argument("--buffer-size", type=int, default=int(1e6))

    # running parameters
    parser.add_argument("--train", action="store_true", default=True)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--n-steps", type=int, default=int(1e6))
    parser.add_argument("--start-learning", type=int, default=int(5e3))
    parser.add_argument("--update-interval", type=int, default=1)
    # UTD
    parser.add_argument("--updates-per-step", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)                          # mini-batch size
    parser.add_argument("--eval-interval", type=int, default=int(1e3))
    parser.add_argument("--eval-n-episodes", type=int, default=10)
    parser.add_argument("--test-n-episodes", type=int, default=int(1e3))
    parser.add_argument("--save-interval", type=int, default=int(1e4))
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    algo_yml_path = "./config/{}/{}.yml".format(args.env, args.env_name.split("-v")[0])
    algo_yml = yaml.load(open(algo_yml_path, 'r'), Loader=yaml.FullLoader)
    for key, value in algo_yml.items():
        setattr(args, key, value)

    setproctitle.setproctitle("{} {}".format(args.algo.upper(), args.env_name))

    seed_start, seed_end = 0, 10
    if not args.train and not args.test:
        raise ValueError("Argument 'train' and 'test' can't be both False")

    """ main function """
    for seed in range(seed_start, seed_end):
        args.seed = seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        # set seed of torch
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        args.stage = "train" if args.train else "test"
        if "ac" in args.algo: runner_id = f"ac-{args.stage}"
        runner = RUNNER[runner_id](args)
        runner.run()

if __name__ == "__main__":
    main()
