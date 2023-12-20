import numpy as np

class ReplayBuffer:
    """ replay buffer """
    def __init__(self, buffer_size, obs_shape, action_dim, extra_info=False, extra_dim=None):
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.memory = {
            "s":    np.zeros((buffer_size, *self.obs_shape), dtype=np.float32),
            "a":    np.zeros((buffer_size, self.action_dim), dtype=np.float32),
            "r":    np.zeros((buffer_size, 1), dtype=np.float32),
            "s_":   np.zeros((buffer_size, *self.obs_shape), dtype=np.float32),
            "done": np.zeros((buffer_size, 1), dtype=np.float32),
        }
        
        self.extra_info = extra_info
        self.extra_dim = extra_dim
        if self.extra_info:
            self.memory["ex"] = np.zeros((buffer_size, self.extra_dim), dtype=np.float32)
            self.memory["ex_"] = np.zeros((buffer_size, self.extra_dim), dtype=np.float32)

        self.capacity = buffer_size
        self.size = 0
        self.cnt = 0

    def store(self, s, a, r, s_, done, ex=None, ex_=None):
        """ store transition (s, a, r, s_, done) """
        self.memory["s"][self.cnt] = s
        self.memory["a"][self.cnt] = a
        self.memory["r"][self.cnt] = r
        self.memory["s_"][self.cnt] = s_
        self.memory["done"][self.cnt] = done
        
        if self.extra_info:
            self.memory["ex"][self.cnt] = ex
            self.memory["ex_"][self.cnt] = ex_

        self.cnt = (self.cnt+1)%self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self, batch_size):
        """ sample a batch of transitions """
        indices = np.random.randint(0, self.size, batch_size)
        batch = {
            "s":    self.memory["s"][indices].copy(),
            "a":    self.memory["a"][indices].copy(),
            "r":    self.memory["r"][indices].copy(),
            "s_":   self.memory["s_"][indices].copy(),
            "done": self.memory["done"][indices].copy()
        }
        
        if self.extra_info:
            batch["ex"] = self.memory["ex"][indices].copy()
            batch["ex_"] = self.memory["ex_"][indices].copy()
        
        return batch

    def clear(self):
        self.size = 0
        self.cnt = 0

class EpisodicReplayBuffer(ReplayBuffer):
    """ episodic replay buffer """
    def __init__(self, buffer_size, obs_shape, action_dim, episode_limit, extra_info=False, extra_dim=None):
        super(EpisodicReplayBuffer, self).__init__(buffer_size, obs_shape, action_dim, extra_info, extra_dim)
        self.cur_start = 0
        self.episodic_starts = []
        self.episodic_lengths = []
        self.episodic_rewards = []
        self.episodic_masks = []
        self.episode_limit = episode_limit

    def store(self, s, a, r, s_, done, timeout, ex=None, ex_=None):
        super(EpisodicReplayBuffer, self).store(s, a, r, s_, done, ex, ex_)
        if done or timeout:
            self.episodic_starts.append(self.cur_start)
            self.episodic_lengths.append((self.cnt - self.cur_start)%self.capacity)
            self.episodic_rewards.append(r)
            mask = np.zeros(self.episode_limit, dtype=np.float32)
            mask[:self.episodic_lengths[-1]] = 1
            self.episodic_masks.append(mask)
            self.cur_start = self.cnt

        if len(self.episodic_starts) > 0 and self.cnt == self.episodic_starts[0]:
            self.episodic_starts.pop(0)
            self.episodic_lengths.pop(0)
            self.episodic_rewards.pop(0)
            self.episodic_masks.pop(0)

    def sample_episode(self, batch_size, reward_priority=False):
        """ sample a batch of episodes """
        assert len(self.episodic_starts) > 0, "no full episode can be sampled "
        epi_starts = np.array(self.episodic_starts, dtype=np.int64)
        epi_lengths = np.array(self.episodic_lengths, dtype=np.int64)
        epi_rewards = np.array(self.episodic_rewards, dtype=np.int64)
        epi_probs = np.exp(epi_rewards) if reward_priority else epi_lengths
        epi_choices = np.random.choice(np.arange(len(epi_starts)), p=epi_probs/epi_probs.sum(), size=batch_size)

        epi_start_indices = epi_starts[epi_choices]
        max_epi_length = max(epi_lengths[epi_choices])
        sample_indices = (epi_start_indices.reshape(-1, 1) + np.arange(max_epi_length))%self.size

        epi_masks = np.vstack(self.episodic_masks)[epi_choices, :max_epi_length, None]
        epi_obs_masks = epi_masks.repeat(np.prod(self.obs_shape), axis=-1).reshape((batch_size, max_epi_length, *self.obs_shape))

        sampled_s = self.memory["s"][sample_indices]*epi_obs_masks
        sampled_a = self.memory["a"][sample_indices]*epi_masks.repeat(self.action_dim, axis=-1)
        sampled_r = self.memory["r"][sample_indices]*epi_masks
        sampled_s_ = self.memory["s_"][sample_indices]*epi_obs_masks
        sampled_done = self.memory["done"][sample_indices]*epi_masks

        # sample
        batch = {
            "s":    sampled_s.copy(),
            "a":    sampled_a.copy(),
            "r":    sampled_r.copy(),
            "s_":   sampled_s_.copy(),
            "done": sampled_done.copy(),
            "mask": epi_masks.copy()
        }
        
        if self.extra_info:
            batch["ex"] = self.memory["ex"][sample_indices]*epi_masks.repeat(self.extra_dim, axis=-1)
            batch["ex_"] = self.memory["ex_"][sample_indices]*epi_masks.repeat(self.extra_dim, axis=-1)
            
        return batch
