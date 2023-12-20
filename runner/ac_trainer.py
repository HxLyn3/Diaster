import os
import json
import numpy as np
from tqdm import tqdm

from env import ENV
from agent import AGENT
from utils import BUFFER

class ACTrainer:
    def __init__(self, args):
        # init env
        self.env = ENV[args.env](args.env_name)
        self.env.reset(seed=args.seed)
        self.env.action_space.seed(args.seed)

        self.eval_env = ENV[args.env](args.env_name)
        self.eval_env.reset(seed=args.seed)
        self.eval_env.action_space.seed(args.seed)

        args.obs_shape = self.env.observation_space.shape
        args.action_space = self.env.action_space
        args.action_dim = int(np.prod(args.action_space.shape))

        # init agent
        self.agent = AGENT[args.algo](
            obs_shape=args.obs_shape,
            hidden_dims=args.hidden_dims,
            action_space=args.action_space,
            return_scale=args.return_scale,
            return_shift=args.return_shift,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            epired_lr=args.epired_lr,
            alpha_lr=args.alpha_lr,
            tau=args.tau,
            gamma=args.gamma,
            device=args.device
        )
        self.agent.train()

        # init replay buffer
        self.memory = BUFFER["episodic"](
            buffer_size=args.buffer_size, 
            obs_shape=args.obs_shape, 
            action_dim=args.action_dim, 
            episode_limit=self.env._max_episode_steps
        )
        
        # running parameters
        self.n_steps = args.n_steps
        self.start_learning = args.start_learning
        self.epired = args.epired
        self.epired_interval = args.epired_interval
        self.epired_batch_size = args.epired_batch_size
        self.update_interval = args.update_interval
        self.updates_per_step = args.updates_per_step
        self.batch_size = args.batch_size
        self.eval_interval = args.eval_interval
        self.eval_n_episodes = args.eval_n_episodes
        self.save_interval = args.save_interval
        self.render = args.render
        self.device = args.device
        self.seed = args.seed
        self.args = args

        self.model_dir = "./result/{}/{}/{}/model".format(args.env, args.env_name, args.algo)
        self.record_dir = "./result/{}/{}/{}/record".format(args.env, args.env_name, args.algo)
        if not os.path.exists(self.model_dir): 
            os.makedirs(self.model_dir)
        if not os.path.exists(self.record_dir):
            os.makedirs(self.record_dir)

    def _warm_up(self):
        """ randomly sample a lot of transitions into buffer before starting learning """
        obs, _ = self.env.reset()

        # step for {self.start_learning} time-steps
        pbar = tqdm(range(self.start_learning), desc="Warming up")
        for _ in pbar:
            action = self.env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.memory.store(obs, action, reward, next_obs, terminated, truncated)

            obs = next_obs
            if done: obs, _ = self.env.reset()

        return obs

    def _eval_policy(self):
        """ evaluate policy """
        episode_rewards = []
        for _ in range(self.eval_n_episodes):
            done = False
            episode_rewards.append(0)
            obs, _ = self.eval_env.reset()
            while not done:
                action = self.agent.act(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                if self.args.env == "pointmaze": reward = int(info["success"])
                episode_rewards[-1] += reward
        return episode_rewards

    def _save(self, records):
        """ save model and record """
        self.agent.save_model(os.path.join(self.model_dir, "model_seed-{}.pth".format(self.seed)))
        with open(os.path.join(self.record_dir, "record_seed-{}.txt".format(self.seed)), "w") as f:
            json.dump(records, f)

    def run(self):
        """ train {args.algo} on {args.env} for {args.n_steps} steps"""

        records = {"step": [], "loss": {"actor": [], "critic": [], "epired": [], "steprew": []}, 
            "alpha": [], "reward_mean": [], "reward_std": [], "reward_min": [], "reward_max": [],}
        obs = self._warm_up()

        actor_loss, critic_loss, eval_reward, alpha = [None]*4
        pbar = tqdm(range(self.n_steps), desc="Training {} on {}.{} (seed: {})".format(
            self.args.algo.upper(), self.args.env.title(), self.args.env_name, self.seed))

        for it in pbar:
            # step in env
            action = self.agent.act(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.memory.store(obs, action, reward, next_obs, terminated, truncated)

            obs = next_obs
            if done: obs, _ = self.env.reset()

            # render
            if self.render: self.env.render()

            # epired
            if self.epired and it % self.epired_interval == 0:
                episodes = self.memory.sample_episode(self.epired_batch_size)
                del episodes["s_"], episodes["done"]
                epired_loss = self.agent.learn_sub_reward_from(**episodes)
                steprew_loss = self.agent.learn_step_reward_from(**episodes)

            # update policy
            if it % self.update_interval == 0:
                for _ in range(int(self.update_interval*self.updates_per_step)):
                    transitions = self.memory.sample(self.batch_size)
                    transitions.pop("r")
                    learning_info = self.agent.learn_policy_from(**transitions)
                    critic_loss = learning_info["loss"]["critic"]
                    actor_loss = learning_info["loss"]["actor"]
                    alpha = learning_info["alpha"]

            # evaluate policy
            if it % self.eval_interval == 0:
                episode_rewards = self._eval_policy()
                records["step"].append(it + self.start_learning)
                records["loss"]["epired"].append(epired_loss)
                records["loss"]["steprew"].append(steprew_loss)
                records["loss"]["critic"].append(critic_loss)
                records["loss"]["actor"].append(actor_loss)
                records["alpha"].append(alpha)
                records["reward_mean"].append(float(np.mean(episode_rewards)))
                records["reward_std"].append(float(np.std(episode_rewards)))
                records["reward_min"].append(float(np.min(episode_rewards)))
                records["reward_max"].append(float(np.max(episode_rewards)))
                eval_reward = records["reward_mean"][-1]

            pbar.set_postfix(
                alpha=alpha,
                actor_loss=actor_loss, 
                critic_loss=critic_loss,
                epired_loss=epired_loss,
                steprew_loss=steprew_loss,
                eval_reward=eval_reward
            )

            # save
            if it % self.save_interval == 0: self._save(records)

        self._save(records)
