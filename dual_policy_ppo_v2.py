"""
双策略PPO V2：自适应好奇心-信任域 (AC-TR)
核心创新：根据好奇心动态调节信任域约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Tuple, Dict, Optional

# 复用 V1 的网络结构
from dual_policy_ppo import ActorCriticNetwork, CuriosityModule

class DualPolicyPPO_V2:
    """
    双策略PPO V2 (AC-TR)
    创新点：Adaptive Curiosity-Driven Trust Region
    当好奇心高时，放宽KL约束；当好奇心低时，收紧KL约束。
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 continuous: bool = False,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 intrinsic_coef: float = 0.1,
                 kl_coef: float = 0.01,
                 curiosity_alpha: float = 10.0,  # 新增：控制好奇心对KL调节的力度
                 update_opponent_interval: int = 10,
                 hidden_dim: int = 256,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.intrinsic_coef = intrinsic_coef
        self.kl_coef = kl_coef
        self.curiosity_alpha = curiosity_alpha
        self.update_opponent_interval = update_opponent_interval
        self.device = device
        
        # 主策略网络
        self.actor_critic = ActorCriticNetwork(
            state_dim, action_dim, hidden_dim, continuous
        ).to(device)
        
        # 对手策略网络（保守的锚点）
        self.opponent_actor_critic = ActorCriticNetwork(
            state_dim, action_dim, hidden_dim, continuous
        ).to(device)
        
        # 初始化对手策略为主策略的副本
        self.opponent_actor_critic.load_state_dict(self.actor_critic.state_dict())
        
        # 好奇心模块
        self.curiosity = CuriosityModule(
            state_dim, action_dim, hidden_dim, continuous
        ).to(device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            list(self.actor_critic.parameters()) + list(self.curiosity.parameters()),
            lr=lr
        )
        
        # 训练统计
        self.update_count = 0
        self.performance_history = []
        
    def compute_kl_divergence(self, states: torch.Tensor) -> torch.Tensor:
        """
        计算主策略和对手策略之间的KL散度 (样本级)
        """
        with torch.no_grad():
            dist_opponent, _ = self.opponent_actor_critic(states)
        
        dist_actor, _ = self.actor_critic(states)
        
        # 计算KL(π_actor || π_opponent)
        if self.continuous:
            kl = torch.distributions.kl_divergence(dist_actor, dist_opponent).sum(dim=-1)
        else:
            kl = torch.distributions.kl_divergence(dist_actor, dist_opponent)
        
        return kl
    
    def compute_intrinsic_reward(self, 
                                  states: torch.Tensor,
                                  actions: torch.Tensor,
                                  next_states: torch.Tensor) -> torch.Tensor:
        """
        计算纯粹的好奇心奖励 (不再乘以 KL 惩罚)
        """
        # 好奇心奖励
        curiosity_reward, _, _ = self.curiosity(states, actions, next_states)
        return curiosity_reward.detach()
    
    def compute_gae(self, 
                    rewards: torch.Tensor,
                    values: torch.Tensor,
                    dones: torch.Tensor,
                    next_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算广义优势估计(GAE)
        """
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, memory: Dict[str, torch.Tensor], 
               batch_size: int = 64, 
               n_epochs: int = 10) -> Dict[str, float]:
        """
        更新策略和价值网络 (AC-TR核心逻辑)
        """
        # 准备数据
        states = memory['states']
        actions = memory['actions']
        old_log_probs = memory['log_probs']
        rewards = memory['rewards']
        dones = memory['dones']
        values = memory['values']
        next_states = memory['next_states']
        
        # 计算内在奖励
        with torch.no_grad():
            intrinsic_rewards = self.compute_intrinsic_reward(states, actions, next_states)
            total_rewards = rewards + self.intrinsic_coef * intrinsic_rewards
        
        # 计算GAE
        with torch.no_grad():
            _, _, next_values = self.actor_critic.get_action(next_states)
            next_values = next_values.squeeze()
            advantages, returns = self.compute_gae(total_rewards, values, dones, next_values)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 训练多个epoch
        n_samples = len(states)
        indices = np.arange(n_samples)
        
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        total_adaptive_kl_coef = 0 # 统计平均KL系数
        total_curiosity_loss = 0
        
        for epoch in range(n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_next_states = next_states[batch_indices]
                
                # 评估当前策略
                log_probs, state_values, entropy = self.actor_critic.evaluate_actions(
                    batch_states, batch_actions
                )
                state_values = state_values.squeeze()
                
                # PPO损失
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = F.mse_loss(state_values, batch_returns)
                
                # AC-TR 核心逻辑：自适应KL约束
                # 1. 计算KL散度
                kl_div = self.compute_kl_divergence(batch_states)
                
                # 2. 计算当前batch的好奇心 (用于调节KL)
                # 注意：这里我们需要再次计算好奇心，或者复用之前计算的
                # 为了简单，我们直接复用 forward 计算的 loss 作为 curiosity intensity
                curiosity_reward_batch, forward_loss, inverse_loss = self.curiosity(
                    batch_states, batch_actions, batch_next_states
                )
                curiosity_loss = forward_loss + inverse_loss
                
                # 3. 动态计算 KL 系数
                # 归一化 curiosity 以防止系数过小
                # 简单的 AC-TR 公式: beta / (1 + alpha * curiosity)
                adaptive_kl_coef = self.kl_coef / (1.0 + self.curiosity_alpha * curiosity_reward_batch.detach())
                
                # 4. 加权 KL Loss
                kl_loss = (adaptive_kl_coef * kl_div).mean()
                
                # 总损失
                loss = (policy_loss + 
                       self.value_coef * value_loss - 
                       self.entropy_coef * entropy.mean() +
                       kl_loss +  # 已经是加权后的
                       curiosity_loss)
                
                # 更新网络
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor_critic.parameters()) + list(self.curiosity.parameters()),
                    max_norm=0.5
                )
                self.optimizer.step()
                
                # 记录统计
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_kl += kl_div.mean().item() # 记录原始KL
                total_adaptive_kl_coef += adaptive_kl_coef.mean().item()
                total_curiosity_loss += curiosity_loss.item()
        
        n_updates = (n_samples // batch_size) * n_epochs
        
        self.update_count += 1
        
        # 返回训练统计
        stats = {
            'loss': total_loss / n_updates,
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'kl_divergence': total_kl / n_updates,
            'adaptive_kl_coef': total_adaptive_kl_coef / n_updates, # 新指标
            'curiosity_loss': total_curiosity_loss / n_updates,
            'intrinsic_reward_mean': intrinsic_rewards.mean().item(),
            'intrinsic_reward_std': intrinsic_rewards.std().item(),
        }
        
        return stats
    
    def update_opponent_policy(self, current_performance: float):
        """
        更新对手策略
        """
        self.performance_history.append(current_performance)
        
        if self.update_count % self.update_opponent_interval == 0:
            if len(self.performance_history) >= self.update_opponent_interval:
                recent_perf = self.performance_history[-self.update_opponent_interval:]
                if all(recent_perf[i] <= recent_perf[i+1] for i in range(len(recent_perf)-1)):
                    self.opponent_actor_critic.load_state_dict(self.actor_critic.state_dict())
                    print(f"✓ 对手策略已更新 (更新次数: {self.update_count})")
                    return True
        
        return False
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'opponent_actor_critic': self.opponent_actor_critic.state_dict(),
            'curiosity': self.curiosity.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'performance_history': self.performance_history,
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.opponent_actor_critic.load_state_dict(checkpoint['opponent_actor_critic'])
        self.curiosity.load_state_dict(checkpoint['curiosity'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.update_count = checkpoint['update_count']
        self.performance_history = checkpoint['performance_history']

