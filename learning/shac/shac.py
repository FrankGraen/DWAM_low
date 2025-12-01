from typing import Any, Mapping, Optional, Tuple, Union
import os
import copy
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from skrl.agents.torch import Agent
from skrl.memories.torch import RandomMemory
from skrl.models.torch import Model
import gym
import gymnasium
from torch.nn.utils.clip_grad import clip_grad_norm_
from learning.shac.utils.running_mean_std import RunningMeanStd
from learning.shac.utils.dataset import CriticDataset
from learning.shac.utils.average_meter import AverageMeter
import learning.shac.utils.torch_utils as tu
import time
import yaml
# from torchviz import make_dot
class SHAC(Agent):
    def __init__(self,
                 models: Mapping[str, Model],
                 memory: Optional[Union[RandomMemory, Tuple[RandomMemory]]] = None,
                 num_envs: int = 1,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Short-Horizon Actor-Critic (SHAC) agent for skrl framework"""
        _cfg = {
            "rollouts": 32,
            "mini_batches": 4,
            "discount_factor": 0.99,
            "lambda": 0.95,
            "actor_learning_rate": 2e-4,  
            "critic_learning_rate": 5e-4,
            "target_critic_alpha": 0.4,
            "obs_rms": False,
            "ret_rms": False,
            "rew_scale": 10.0,
            "critic_iterations": 4,
            "truncate_grads": True,
            "grad_norm": 1.0,
            "max_epochs": 1000000 / 32,
            "betas": (0.9, 0.95),
            "max_episode_length": 40 * 5,  
            "experiment": {
                "directory": "PushBox",
                "experiment_name": "",
                "write_interval": 40,
                "checkpoint_interval": 400,
                "store_separately": False,
                "wandb": True,
                "wandb_kwargs": {}
            }
        }
        _cfg.update(cfg.get("agent", {}) if cfg else {})
        super().__init__(models, memory, observation_space, action_space, device, cfg=_cfg)

        # Models
        self.actor = self.models.get("policy")
        self.critic = self.models.get("value")
        self.target_critic = copy.deepcopy(self.critic)


        # Configuration
        self.num_obs = observation_space.shape[0]
        self.num_actions = action_space.shape[0]
        self.num_envs = num_envs
        self.rollouts = self.cfg["rollouts"]
        self.gamma = self.cfg["discount_factor"]
        self.lam = self.cfg.get("lambda", 0.95)
        self.actor_lr = self.cfg["actor_learning_rate"]
        self.critic_lr = self.cfg["critic_learning_rate"]
        self.target_critic_alpha = self.cfg["target_critic_alpha"]
        self.critic_iterations = self.cfg["critic_iterations"]
        self.num_batch = self.cfg["mini_batches"]
        self.batch_size = self.num_envs * self.rollouts // self.num_batch
        self.truncate_grad = self.cfg["truncate_grads"]
        self.grad_norm = self.cfg["grad_norm"]
        self.rew_scale = self.cfg["rew_scale"]
        self.betas = self.cfg["betas"]
        self.max_episode_length = self.cfg["max_episode_length"]

        # Normalization
        self.obs_rms = RunningMeanStd(shape=(self.num_obs,), device=self.device) if self.cfg["obs_rms"] else None
        self.ret_rms = RunningMeanStd(shape=(), device=self.device) if self.cfg["ret_rms"] else None
        self.ret = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr, betas=self.betas)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, betas=self.betas)

        self.obs_buf = torch.zeros((self.rollouts, self.num_envs, self.num_obs), dtype=torch.float32, device=self.device)
        self.rew_buf_with_grad = torch.zeros((self.rollouts, self.num_envs, 1), dtype=torch.float32, device=self.device)
        self.rew_buf = torch.zeros((self.rollouts, self.num_envs, 1), dtype=torch.float32, device=self.device)
        self.done_mask = torch.zeros((self.rollouts, self.num_envs, 1), dtype=torch.float32, device=self.device)
        self.next_values_buf = torch.zeros((self.rollouts, self.num_envs, 1), dtype=torch.float32, device=self.device)
        self.record_next_values = torch.zeros((self.rollouts + 1, self.num_envs, 1), dtype=torch.float32, device=self.device)

        self.current_step = 0

        # Logging
        self.log_dir = os.path.join(self.cfg["experiment"]["directory"], self.cfg["experiment"]["experiment_name"])
        if self.cfg["experiment"]["directory"]:
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.log_dir, "log"))
            yaml.dump(self.cfg, open(os.path.join(self.log_dir, "cfg.yaml"), "w"))
        self.save_interval = self.cfg["experiment"]["checkpoint_interval"]
        self.iter_count = 0
        self.step_count = 0
        self.best_policy_loss = np.inf
        self.actor_loss_log = np.inf
        self.value_loss_log_log = np.inf
        self.episode_loss_his = []
        self.episode_discounted_loss_his = []
        self.episode_length_his = []
        self.episode_loss = torch.zeros(self.num_envs, device=self.device)
        self.episode_discounted_loss = torch.zeros(self.num_envs, device=self.device)
        self.episode_gamma = torch.ones(self.num_envs, device=self.device)
        self.episode_length = torch.zeros(self.num_envs, dtype=int, device=self.device)
        self.episode_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_discounted_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)

        self.rollout_count = 0

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        super().init(trainer_cfg=trainer_cfg)
        print(f"\033[33;1mSHAC agent initialized with {self.num_envs} environments\033[0m")
        self.set_mode("train") 

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> Tuple[torch.Tensor, Any, Any]:
        with torch.no_grad():
            if self.obs_rms:
                self.obs_rms.update(states)
                states = self.obs_rms.normalize(states)
        actions, log_prob, outputs = self.actor.act({"states": states.clone().detach()}, role="actor")
        self.current_log_prob = log_prob
        return actions, log_prob, outputs

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        pass

    def record_transition(self,
                         states: torch.Tensor,
                         actions: torch.Tensor,
                         rewards: torch.Tensor,
                         next_states: torch.Tensor,
                         terminated: torch.Tensor,
                         truncated: torch.Tensor,
                         infos: Any,
                         timestep: int,
                         timesteps: int) -> None:
        # print(f"record_transition: states.shape={states.shape}, actions.shape={actions.shape}, "
        #       f"rewards.shape={rewards.shape}, next_states.shape={next_states.shape}, "
        #       f"terminated.shape={terminated.shape}")
        step_idx = self.current_step % self.rollouts
        record_start_time = time.time()
        with torch.no_grad():
            if self.obs_rms is not None:
                obs_rms = copy.deepcopy(self.obs_rms)
            if self.ret_rms is not None:
                ret_var = self.ret_rms.var.clone()
            raw_rewards = rewards.clone()

        rewards = rewards * self.rew_scale
        if self.obs_rms is not None:
            with torch.no_grad():
                self.obs_rms.update(next_states.squeeze(-1))
            next_states = self.obs_rms.normalize(next_states)
        if self.ret_rms is not None:
            with torch.no_grad():
                self.ret = self.ret * self.gamma + rewards.squeeze(-1)
                self.ret_rms.update(self.ret)
            rewards = rewards / torch.sqrt(ret_var + 1e-6)

        self.episode_length += 1
        done = terminated | truncated  # [64, 1]
        done_env_ids = done.nonzero(as_tuple=False)[:, 0]

        self.record_next_values[step_idx + 1] = self.target_critic.act({"states": next_states}, role="value")[0]  # [64, 1]

        for id in done_env_ids:
            id_scalar = id.item()
            if torch.isnan(states[id_scalar]).sum() > 0 or torch.isinf(states[id_scalar]).sum() > 0 or (torch.abs(next_states[id_scalar]) > 1e6).sum() > 0:
                self.record_next_values[step_idx + 1,id_scalar] = 0.0
            elif self.episode_length[id_scalar] < self.max_episode_length:
                self.record_next_values[step_idx + 1,id_scalar] = 0.0
            else:
                if self.obs_rms is not None:
                    real_obs = obs_rms.normalize(states[id_scalar])
                else:
                    real_obs = states[id_scalar]
                self.record_next_values[step_idx + 1,id_scalar] = self.target_critic.act({"states": real_obs.unsqueeze(0)}, role="value")[0].squeeze()

        if (self.record_next_values[step_idx + 1].abs() > 1e6).any():
            print("\033[31;1mnext_values error\033[0m")
            raise ValueError
        
        self.rew_buf_with_grad[step_idx] = rewards.clone()
        with torch.no_grad():
            self.rew_buf[step_idx] = rewards.clone()
            self.obs_buf[step_idx] = states.clone()
            if step_idx < self.rollouts - 1:
                self.done_mask[step_idx] = done.clone().to(torch.float32)
            else:
                self.done_mask[step_idx, :] = 1.
            self.next_values_buf[step_idx] = self.record_next_values[step_idx + 1].clone()


        with torch.no_grad():
            self.episode_loss -= raw_rewards.squeeze(-1)
            self.episode_discounted_loss -= self.episode_gamma * raw_rewards.squeeze(-1)
            self.episode_gamma *= self.gamma

            if len(done_env_ids) > 0:
                self.episode_loss_meter.update(self.episode_loss[done_env_ids])
                self.episode_discounted_loss_meter.update(self.episode_discounted_loss[done_env_ids])
                self.episode_length_meter.update(self.episode_length[done_env_ids])
                for done_env_id in done_env_ids:
                    done_env_id_scalar = done_env_id.item()
                    if self.episode_loss[done_env_id_scalar].abs() > 1e6:
                        print("\033[31;1mep loss error\033[0m")
                        raise ValueError
                    self.episode_loss_his.append(self.episode_loss[done_env_id_scalar].item())
                    self.episode_discounted_loss_his.append(self.episode_discounted_loss[done_env_id_scalar].item())
                    self.episode_length_his.append(self.episode_length[done_env_id_scalar].item())
                    self.episode_loss[done_env_id_scalar] = 0.0
                    self.episode_discounted_loss[done_env_id_scalar] = 0.0
                    self.episode_length[done_env_id_scalar] = 0
                    self.episode_gamma[done_env_id_scalar] = 1.0

        self.current_step += 1
        record_end_time = time.time()
        # print(f"\033[36;1mRecorded transition at step {self.current_step} in {record_end_time - record_start_time:.4f} seconds\033[0m")

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        self.rollout_count += 1
        if self.rollout_count >= self.rollouts:
            # self.set_mode("train")
            self._update(timestep, timesteps)
            # self.set_mode("eval")
            self.rollout_count = 0

            self.obs_buf = torch.zeros((self.rollouts, self.num_envs, self.num_obs), dtype=torch.float32, device=self.device)
            self.rew_buf = torch.zeros((self.rollouts, self.num_envs, 1), dtype=torch.float32, device=self.device)
            self.rew_buf_with_grad = torch.zeros((self.rollouts, self.num_envs, 1), dtype=torch.float32, device=self.device)
            self.done_mask = torch.zeros((self.rollouts, self.num_envs, 1), dtype=torch.float32, device=self.device)
            self.next_values_buf = torch.zeros((self.rollouts, self.num_envs, 1), dtype=torch.float32, device=self.device)
            self.record_next_values = torch.zeros((self.rollouts + 1, self.num_envs, 1), dtype=torch.float32, device=self.device)
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        time_start_epoch = time.time()
        # torch.autograd.set_detect_anomaly(True)
        # print(f"\033[33;1mreward min: {self.rew_buf.min().item():.3f}, reward max: {self.rew_buf.max().item():.3f}\033[0m")
        # Actor update
        def actor_closure(): 
            # print(f"\033[33;1mactor_closure called at step {self.current_step}\033[0m")
            actor_loss = torch.tensor(0., dtype=torch.float32, device=self.device)
            rew_acc = torch.zeros((self.rollouts + 1, self.num_envs, 1), dtype=torch.float32, device=self.device)
            gamma = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
            cycle_start_time = time.time()
            for step_idx in range(self.rollouts):
                next_values = self.record_next_values[step_idx + 1].clone().detach()
                done_env_ids = (self.done_mask[step_idx].squeeze(-1) > 0).nonzero(as_tuple=False)[:, 0]

                rew_acc[step_idx + 1] = rew_acc[step_idx] + gamma.unsqueeze(-1) * self.rew_buf_with_grad[step_idx]
                if step_idx < self.rollouts - 1:
                    actor_loss = actor_loss + (-rew_acc[step_idx + 1, done_env_ids] - self.gamma * gamma[done_env_ids].unsqueeze(-1) * next_values[done_env_ids]).sum()
                else:
                    actor_loss = actor_loss + (-rew_acc[step_idx + 1] - self.gamma * gamma.unsqueeze(-1) * next_values).sum()

                gamma = gamma * self.gamma
                gamma[done_env_ids] = 1.0
                rew_acc[step_idx + 1, done_env_ids] = 0.0
            cycle_end_time = time.time()
            # print(f"\033[35;1mActor loss cycle took {cycle_end_time - cycle_start_time:.4f} seconds\033[0m")
            # if self.iter_count == 0:
            #     os.makedirs(self.log_dir, exist_ok=True)
            #     graph = make_dot(actor_loss, params=dict(self.actor.named_parameters()), show_attrs=True, show_saved=True)
            #     graph.render(os.path.join(self.log_dir, f"actor_loss_graph_step"), format="png", cleanup=True)
            #     print("actor_loss graph saved")

            actor_loss = actor_loss / (self.rollouts * self.num_envs)
            if self.ret_rms is not None:
                with torch.no_grad():
                    ret_var = self.ret_rms.var.clone()
                actor_loss *= torch.sqrt(ret_var + 1e-6)
            self.actor_loss_log = actor_loss.detach().cpu().item()
            backward_start_time = time.time()
            actor_loss.backward()
            backward_end_time = time.time()
            # print(f"\033[35;1mActor loss backward took {backward_end_time - backward_start_time:.4f} seconds\033[0m")
            with torch.no_grad():
                grads = [p.grad for p in self.actor.parameters() if p.grad is not None]
                if not grads:
                    print("\033[31;1mNo valid gradients found\033[0m")
                self.grad_norm_before_clip = tu.grad_norm(self.actor.parameters(), device=self.device)
                if self.truncate_grad:
                    clip_grad_norm_(self.actor.parameters(), self.grad_norm)
                self.grad_norm_after_clip = tu.grad_norm(self.actor.parameters(), device=self.device)
                if torch.isnan(self.grad_norm_before_clip) or self.grad_norm_before_clip > 1000000.0:
                    print("\033[31;1mNaN gradient\033[0m")
                    raise ValueError
            self.actor_optimizer.zero_grad()
            return actor_loss
        actor_update_start_time = time.time()
        self.actor_optimizer.step(actor_closure).detach().item()
        actor_update_end_time = time.time()
        print(f"\033[35;1mActor update took {actor_update_end_time - actor_update_start_time:.4f} seconds\033[0m")
        
        compute_target_values_start_time = time.time()
        # Compute target_values
        with torch.no_grad():
            target_values = torch.zeros_like(self.rew_buf)  # [32, 64, 1]
            if self.lam is not None:  # TD(lambda)
                Ai = torch.zeros(self.num_envs, device=self.device)
                Bi = torch.zeros(self.num_envs, device=self.device)
                lam = torch.ones(self.num_envs, device=self.device)
                for i in reversed(range(self.rollouts)):
                    lam = lam * self.lam * (1.0 - self.done_mask[i].squeeze(-1)) + self.done_mask[i].squeeze(-1)
                    Ai = (1.0 - self.done_mask[i].squeeze(-1)) * (
                        self.lam * self.gamma * Ai + self.gamma * self.next_values_buf[i].squeeze(-1) +
                        (1.0 - lam) / (1.0 - self.lam) * self.rew_buf[i].squeeze(-1)
                    )
                    Bi = self.gamma * (
                        self.next_values_buf[i].squeeze(-1) * self.done_mask[i].squeeze(-1) +
                        Bi * (1.0 - self.done_mask[i].squeeze(-1))
                    ) + self.rew_buf[i].squeeze(-1)
                    target_values[i] = ((1.0 - self.lam) * Ai + lam * Bi).unsqueeze(-1)
            else:  # One-step
                target_values = self.rew_buf + self.gamma * self.next_values_buf

        compute_target_values_end_time = time.time()
        # print(f"\033[35;1mCompute target values took {compute_target_values_end_time - compute_target_values_start_time:.4f} seconds\033[0m")


        critic_update_start_time = time.time()
        # Critic update
        dataset = CriticDataset(self.batch_size, self.obs_buf, target_values)
        self.value_loss_log = 0.0
        for _ in range(self.critic_iterations):
            total_critic_loss = 0.
            batch_cnt = 0
            for i in range(len(dataset)):
                batch_sample = dataset[i]
                self.critic_optimizer.zero_grad()
                predicted_values, _, _ = self.critic({"states": batch_sample["obs"]})
                predicted_values = predicted_values.squeeze(-1)
                target_values = batch_sample["target_values"]
                critic_loss = ((predicted_values - target_values) ** 2).mean()
                critic_loss.backward()
                for param in self.critic.parameters():
                    param.grad.nan_to_num_(0.0, 0.0, 0.0)
                if self.truncate_grad:
                    clip_grad_norm_(self.critic.parameters(), self.grad_norm)
                #self.critic_optimizer.step()
                total_critic_loss += critic_loss
                batch_cnt += 1
            self.value_loss_log = (total_critic_loss / batch_cnt).detach().cpu().item()
        critic_update_end_time = time.time()
        print(f"\033[35;1mCritic update took {critic_update_end_time - critic_update_start_time:.4f} seconds\033[0m")

        # Update target critic
        with torch.no_grad():
            for param, param_targ in zip(self.critic.parameters(), self.target_critic.parameters()):
                param_targ.data.mul_(self.target_critic_alpha)
                param_targ.data.add_((1.0 - self.target_critic_alpha) * param.data)

        # Logging
        self.iter_count += 1
        time_end_epoch = time.time()
        # if self.episode_loss_his:
        mean_policy_loss = self.episode_loss_meter.get_mean()
        mean_policy_discounted_loss = self.episode_discounted_loss_meter.get_mean()
        mean_episode_length = self.episode_length_meter.get_mean()
        if mean_policy_loss < self.best_policy_loss:
            self.best_policy_loss = mean_policy_loss
        fps = self.rollouts * self.num_envs / (time_end_epoch - time_start_epoch)
        print(f"\033[32;1miter {self.iter_count}: ep loss {mean_policy_loss:.2f}, ep discounted loss {mean_policy_discounted_loss:.2f}, ep len {mean_episode_length:.1f}, fps total {fps:.2f}, value loss {self.value_loss_log:.2f}, grad norm before clip {self.grad_norm_before_clip:.2f}, grad norm after clip {self.grad_norm_after_clip:.2f}\033[0m")
                  
            # if self.writer:
            #     self.writer.add_scalar("actor_loss", self.actor_loss_log, self.step_count)
            #     self.writer.add_scalar("value_loss", self.value_loss_log, self.iter_count)
            #     self.writer.add_scalar("policy_loss", mean_policy_loss, self.step_count)
            #     self.writer.add_scalar("policy_discounted_loss", mean_policy_discounted_loss, self.step_count)
            #     self.writer.add_scalar("episode_length", mean_episode_length, self.step_count)
        # Learning rate schedule
        actor_lr = (1e-5 - self.actor_lr) * (self.iter_count / self.cfg["max_epochs"]) + self.actor_lr
        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = actor_lr
        critic_lr = (1e-5 - self.critic_lr) * (self.iter_count / self.cfg["max_epochs"]) + self.critic_lr
        for param_group in self.critic_optimizer.param_groups:
            param_group["lr"] = critic_lr
        # if self.writer:
        #     self.writer.add_scalar("lr/actor_lr", actor_lr, self.iter_count)
        #     self.writer.add_scalar("lr/critic_lr", critic_lr, self.iter_count)


