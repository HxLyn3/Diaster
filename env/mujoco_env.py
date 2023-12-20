import gymnasium as gym

class MujocoEnv(gym.Wrapper):
    """ mujoco env with only episodic reward """
    def __init__(self, env_name) -> None:
        self.episodic_reward = 0
        self.episodic_ctrl_cost = 0
        env = gym.make(env_name)
        super(MujocoEnv, self).__init__(env)
        env._max_episode_steps = 1000
        self._max_episode_steps = env._max_episode_steps

    def reset(self, **kwargs):
        self.episodic_reward = 0
        self.episodic_ctrl_cost = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self.episodic_reward += reward

        reward = self.episodic_reward if terminated or truncated else 0
        return next_obs, reward, terminated, truncated, info
