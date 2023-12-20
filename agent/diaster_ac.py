import torch
import numpy as np
from copy import deepcopy
from torch.optim import Adam

from component import ACTOR, CRITIC, REWARD

class DiasterACAgent:
    """ SAC with Diaster """
    def __init__(self, **kargs):
        # set arguments
        for key, value in kargs.items():
            setattr(self, key, value)

        # action space
        self.action_dim = np.prod(self.action_space.shape)
        self.max_action = torch.as_tensor(
            (self.action_space.high - self.action_space.low)/2).to(self.device)
        self.target_entropy = -self.action_dim

        # actor
        self.actor = ACTOR["prob"](
            obs_shape=self.obs_shape, 
            hidden_dims=self.hidden_dims, 
            action_dim=self.action_dim, 
            max_action=self.max_action,
            sigma_cond=True
        ).to(self.device)
        
        # critic
        self.critic1 = CRITIC["q"](self.obs_shape, self.hidden_dims, self.action_dim).to(self.device)
        self.critic2 = CRITIC["q"](self.obs_shape, self.hidden_dims, self.action_dim).to(self.device)

        # target critic
        self.critic1_trgt = deepcopy(self.critic1)
        self.critic2_trgt = deepcopy(self.critic2)
        self.critic1_trgt.eval()
        self.critic2_trgt.eval()

        # sub-trajectory reward function
        self.sub_rew = REWARD["rnn"](np.prod(self.obs_shape)+self.action_dim).to(self.device)
        # step-wise reward function
        self.step_rew = REWARD["mlp"](self.obs_shape, self.hidden_dims, self.action_dim).to(self.device)

        # optimizer
        self.actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic1_optim = Adam(list(self.critic1.parameters()), lr=self.critic_lr)
        self.critic2_optim = Adam(list(self.critic2.parameters()), lr=self.critic_lr)
        self.red_optim = Adam(self.sub_rew.parameters(), lr=self.epired_lr)
        self.step_rew_optim = Adam(self.step_rew.parameters(), lr=self.epired_lr)

        # alpha: weight of entropy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.detach().exp()
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)

        self._eps = np.finfo(np.float32).eps.item()

    def train(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()
        self.sub_rew.train()
        self.step_rew.train()

    def eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        self.sub_rew.eval()
        self.step_rew.eval()

    def _sync_weight(self):
        """ synchronize weight """
        for trgt, src in zip(self.critic1_trgt.parameters(), self.critic1.parameters()):
            trgt.data.copy_(trgt.data*(1.0-self.tau) + src.data*self.tau)
        for trgt, src in zip(self.critic2_trgt.parameters(), self.critic2.parameters()):
            trgt.data.copy_(trgt.data*(1.0-self.tau) + src.data*self.tau)

    def actor4ward(self, obs, deterministic=False):
        """ forward propagation of actor """
        dist = self.actor(obs)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action)

        squashed_action = torch.tanh(action)
        log_prob = log_prob - torch.log(self.max_action*(1-squashed_action.pow(2))+self._eps).sum(-1, keepdim=True)

        return self.max_action*squashed_action, log_prob

    def act(self, obs, deterministic=False):
        """ sample action """
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            action, _ = self.actor4ward(obs, deterministic)
            action = action.cpu().detach().numpy()
        return action

    def learn_sub_reward_from(self, s, a, r, mask):
        # shape: (batch_size, time_steps, -1)
        bs, steps = s.shape[:2]
        s    = torch.as_tensor(s, device=self.device).view(bs*steps, -1)
        a    = torch.as_tensor(a, device=self.device).view(bs*steps, -1)
        r    = torch.as_tensor(r, device=self.device).sum(dim=1)/self.return_scale + self.return_shift  # episodic return
        mask = torch.as_tensor(mask, device=self.device).view(bs, steps)
        mask_len = mask.sum(dim=-1).long()

        s_a = torch.cat((s, a), dim=-1).view(bs, steps, -1)

        total_red_loss = 0
        for it in range(steps):
            break_p = np.random.randint(steps)
            self.sub_rew.init_hidden()
            sub_r = self.sub_rew(s_a[:, :break_p+1])*mask[:, :break_p+1]
            if break_p < steps - 1:
                self.sub_rew.init_hidden()
                sub_r2 = self.sub_rew(s_a[:, break_p+1:])*mask[:, break_p+1:]
                sub_r = torch.cat((sub_r, sub_r2), dim=-1)
            sub_r = sub_r[torch.arange(bs), mask_len-1] + (mask_len > break_p+1).float()*sub_r[:, break_p]
            red_loss = ((sub_r.flatten() - r.flatten()).pow(2)).mean()
            self.red_optim.zero_grad()
            red_loss.backward()
            self.red_optim.step()
            total_red_loss += red_loss.item()
        return total_red_loss/it

    def learn_step_reward_from(self, s, a, r, mask):
        # shape: (batch_size, time_steps, -1)
        bs, steps = s.shape[:2]
        s    = torch.as_tensor(s, device=self.device).view(bs*steps, -1)
        a    = torch.as_tensor(a, device=self.device).view(bs*steps, -1)
        r    = torch.as_tensor(r, device=self.device).sum(dim=1)/self.return_scale + self.return_shift  # episodic return
        mask = torch.as_tensor(mask, device=self.device).view(bs, steps)
        mask_len = mask.sum(dim=-1).long()

        s_a = torch.cat((s, a), dim=-1).view(bs, steps, -1)
        with torch.no_grad():
            self.sub_rew.init_hidden()
            sub_r = self.sub_rew(s_a)*mask
            sub_r[torch.arange(bs), mask_len-1] = r.flatten()
            diff_r = sub_r - torch.cat((torch.zeros((bs, 1), device=self.device), sub_r[:, :-1]), dim=-1)

        total_steprewloss = 0
        for it in range(steps):
            indices = np.random.choice(np.arange(bs*steps), size=bs, 
                p=(mask.flatten()/mask.sum()).cpu().numpy(), replace=False)
            step_trgt = diff_r.flatten()[indices]
                
            step_r = self.step_rew(s[indices], a[indices]).flatten()
            steprew_loss = (step_r-step_trgt).pow(2).mean()
            self.step_rew_optim.zero_grad()
            steprew_loss.backward()
            self.step_rew_optim.step()
            total_steprewloss += steprew_loss.item()
        return total_steprewloss/it

    def learn_policy_from(self, s, a, s_, done):
        # shape: (batch_size, -1)
        s    = torch.as_tensor(s, device=self.device)
        a    = torch.as_tensor(a, device=self.device)
        s_   = torch.as_tensor(s_, device=self.device)
        done = torch.as_tensor(done, device=self.device)

        with torch.no_grad():
            step_r = self.step_rew(s, a).flatten()
            a_, log_prob_ = self.actor4ward(s_)
            q_ = torch.min(self.critic1_trgt(s_, a_), self.critic2_trgt(s_, a_)) - self.alpha*log_prob_
            q_trgt = (step_r + self.gamma*(1 - done.flatten())*q_.flatten())

        q1, q2 = self.critic1(s, a).flatten(), self.critic2(s, a).flatten()

        critic1_loss = (q1-q_trgt).pow(2).mean()
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        critic2_loss = (q2-q_trgt).pow(2).mean()
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # update actor
        a, log_prob = self.actor4ward(s)
        q1, q2 = self.critic1(s, a).flatten(), self.critic2(s, a).flatten()
        actor_loss = (self.alpha*log_prob.flatten() - torch.min(q1, q2)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # update alpha
        log_prob = (log_prob.detach() + self.target_entropy).flatten()
        alpha_loss = -(self.log_alpha*log_prob).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.detach().exp()

        # synchronize weight
        self._sync_weight()

        info = {
            "loss": {
                "actor": actor_loss.item(),
                "critic": (critic1_loss.item()+critic2_loss.item())/2,
                "alpha": alpha_loss.item()
            },
            "alpha": self.alpha.item()
        }

        return info

    def save_model(self, filepath):
        """ save model """
        state_dict = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "sub_rew": self.sub_rew.state_dict(),
            "step_rew": self.step_rew.state_dict(),
            "alpha": self.alpha
        }
        torch.save(state_dict, filepath)

    def load_model(self, filepath):
        """ load model """
        state_dict = torch.load(filepath)
        self.actor.load_state_dict(state_dict["actor"])
        self.critic1.load_state_dict(state_dict["critic1"])
        self.critic2.load_state_dict(state_dict["critic2"])
        self.sub_rew.load_state_dict(state_dict["sub_rew"])
        self.step_rew.load_state_dict(state_dict["step_rew"])
        self.alpha = state_dict["alpha"]
