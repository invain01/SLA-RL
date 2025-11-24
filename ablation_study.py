"""
消融实验 (Ablation Study)：对比 Standard Dual PPO 和 Improved PPO

对比算法：
1. Standard Dual PPO：原始双策略PPO（基准1）
2. Improved PPO：改进版（软更新 + KL系数调整）（基准2）
3. No Curiosity：无好奇心模块
4. No KL Constraint：无KL散度约束
5. No Opponent Policy：无对手策略（单策略）
6. No Soft Update：无软更新（使用硬更新）
7. Standard PPO：标准PPO（所有特殊组件都去掉）

目的：量化各个改进组件对性能的贡献
"""

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import torch
import numpy as np
import gym
from dual_policy_ppo import DualPolicyPPO
from trainer import Trainer
import matplotlib.pyplot as plt
import os
from datetime import datetime


# ==================== 改进版 Dual PPO ====================
class ImprovedDualPPO(DualPolicyPPO):
    """
    改进版 Dual PPO（完整版）
    1. 软更新 Opponent (tau=0.01)
    2. 调整 KL 惩罚系数（加底 0.1 + 0.9 * (1 - KL)）
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = 0.01  # 软更新系数

    def compute_intrinsic_reward(self, states, actions, next_states):
        """改进的内在奖励：r_intrinsic = Curiosity * [0.1 + 0.9 * (1 - KL)]"""
        curiosity_reward, _, _ = self.curiosity(states, actions, next_states)
        kl_div = self.compute_kl_divergence(states)
        kl_penalty = torch.clamp(kl_div / self.kl_threshold, 0, 1)
        coefficient = 0.1 + 0.9 * (1 - kl_penalty)  # 改进点：加底
        intrinsic_reward = curiosity_reward * coefficient
        return intrinsic_reward.detach()

    def update(self, memory, batch_size=64, n_epochs=10):
        stats = super().update(memory, batch_size, n_epochs)
        # 改进点：软更新 Opponent
        for param, target_param in zip(self.actor_critic.parameters(), 
                                       self.opponent_actor_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return stats

    def update_opponent_policy(self, current_performance):
        self.performance_history.append(current_performance)
        return False  # 禁用硬更新


# ==================== 消融变体 ====================
class NoCuriosityPPO(ImprovedDualPPO):
    """去除好奇心模块"""
    def compute_intrinsic_reward(self, states, actions, next_states):
        return torch.zeros(len(states), device=self.device)


class NoKLConstraintPPO(ImprovedDualPPO):
    """去除KL散度约束"""
    def __init__(self, *args, **kwargs):
        kwargs['kl_coef'] = 0.0
        super().__init__(*args, **kwargs)
    
    def compute_intrinsic_reward(self, states, actions, next_states):
        """无KL约束，直接返回好奇心奖励"""
        curiosity_reward, _, _ = self.curiosity(states, actions, next_states)
        return curiosity_reward.detach()


class NoOpponentPPO(ImprovedDualPPO):
    """去除对手策略（单策略版本）"""
    def compute_kl_divergence(self, states):
        return torch.zeros(len(states), device=self.device)
    
    def update_opponent_policy(self, current_performance):
        self.performance_history.append(current_performance)
        return False


class NoSoftUpdatePPO(ImprovedDualPPO):
    """去除软更新，使用原始的硬更新逻辑"""
    def update(self, memory, batch_size=64, n_epochs=10):
        # 调用 DualPolicyPPO 的 update（不执行软更新）
        return DualPolicyPPO.update(self, memory, batch_size, n_epochs)
    
    def update_opponent_policy(self, current_performance):
        # 使用原始的硬更新逻辑
        return DualPolicyPPO.update_opponent_policy(self, current_performance)


class StandardPPO(ImprovedDualPPO):
    """标准PPO（所有特殊组件都去除）"""
    def compute_intrinsic_reward(self, states, actions, next_states):
        return torch.zeros(len(states), device=self.device)
    
    def update(self, memory, batch_size=64, n_epochs=10):
        """简化的标准PPO更新"""
        states = memory['states']
        actions = memory['actions']
        old_log_probs = memory['log_probs']
        rewards = memory['rewards']
        dones = memory['dones']
        values = memory['values']
        next_states = memory['next_states']
        
        # 计算GAE（只使用外部奖励）
        with torch.no_grad():
            _, _, next_values = self.actor_critic.get_action(next_states)
            next_values = next_values.squeeze()
            advantages, returns = self.compute_gae(rewards, values, dones, next_values)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 训练循环
        n_samples = len(states)
        indices = np.arange(n_samples)
        
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
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
                value_loss = torch.nn.functional.mse_loss(state_values, batch_returns)
                
                # 总损失（只有策略、价值和熵）
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                
                # 更新
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
        
        n_updates = (n_samples // batch_size) * n_epochs
        
        return {
            'loss': total_loss / n_updates,
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'kl_divergence': 0.0,
            'curiosity_loss': 0.0,
            'intrinsic_reward_mean': 0.0,
            'intrinsic_reward_std': 0.0,
        }


def run_experiment(env_name, agent_class, config_name, config, seed=42):
    """运行单个实验"""
    print(f"\n{'='*70}")
    print(f"消融实验: {config_name}")
    print(f"环境: {env_name}")
    print(f"配置: IC={config['intrinsic_coef']:.4f}, KL={config['kl_coef']:.4f}")
    print(f"{'='*70}")
    
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 获取环境信息
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        continuous = False
    else:
        action_dim = env.action_space.shape[0]
        continuous = True
    
    env.close()
    
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    
    # 创建智能体
    agent = agent_class(
        state_dim=state_dim,
        action_dim=action_dim,
        continuous=continuous,
        lr=config['lr'],
        gamma=config['gamma'],
        intrinsic_coef=config['intrinsic_coef'],
        kl_coef=config['kl_coef'],
        entropy_coef=config['entropy_coef'],
        hidden_dim=config['hidden_dim'],
    )
    
    # 创建保存目录
    safe_name = config_name.replace(' ', '_').replace('(', '').replace(')', '')
    save_dir = f'./ablation_test/{env_name}_{safe_name}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建训练器
    trainer = Trainer(
        env_name=env_name,
        agent=agent,
        max_episodes=config['max_episodes'],
        update_frequency=config['update_frequency'],
        eval_frequency=config['eval_frequency'],
        save_frequency=10000,
        log_frequency=config['log_frequency'],
        save_dir=save_dir
    )
    
    # 训练
    history = trainer.train()
    trainer.close()
    
    return history


def plot_ablation_results(results, save_dir='./ablation_test'):
    """绘制消融实验结果"""
    n_envs = len(results)
    fig = plt.figure(figsize=(20, 6 * n_envs))
    
    # 定义颜色和线型
    colors = {
        'Standard Dual PPO': '#1976D2',        # 蓝色 - 原始版
        'Improved PPO': '#2E7D32',             # 深绿 - 改进版
        'No Curiosity': '#FF6F00',             # 橙色
        'No KL Constraint': '#D32F2F',         # 红色
        'No Opponent Policy': '#9C27B0',       # 紫色
        'No Soft Update': '#FF9800',           # 深橙
        'Standard PPO': '#616161',             # 灰色
    }
    
    linestyles = {
        'Standard Dual PPO': '-',              # 实线
        'Improved PPO': '-',                   # 实线
        'No Curiosity': '--',                  # 虚线
        'No KL Constraint': '--',              # 虚线
        'No Opponent Policy': '--',            # 虚线
        'No Soft Update': '--',                # 虚线
        'Standard PPO': ':',                   # 点线
    }
    
    linewidths = {
        'Standard Dual PPO': 2.5,
        'Improved PPO': 3.0,                   # 改进版加粗
        'No Curiosity': 2.0,
        'No KL Constraint': 2.0,
        'No Opponent Policy': 2.0,
        'No Soft Update': 2.0,
        'Standard PPO': 2.5,
    }
    
    for env_idx, (env_name, env_results) in enumerate(results.items()):
        if not env_results:
            continue
        
        # 1. 训练奖励曲线（平滑）
        ax1 = plt.subplot(n_envs, 4, env_idx * 4 + 1)
        for algo_name, history in env_results.items():
            rewards = history['episode_rewards']
            if len(rewards) > 0:
                window = min(50, max(1, len(rewards) // 10))
                if window > 1:
                    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    x = range(window-1, len(rewards))
                    ax1.plot(x, smoothed, 
                            linewidth=linewidths.get(algo_name, 2.0),
                            label=algo_name, 
                            color=colors.get(algo_name, 'black'), 
                            linestyle=linestyles.get(algo_name, '-'),
                            alpha=0.9)
        
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Reward (Smoothed)', fontsize=12)
        ax1.set_title(f'{env_name} - Training Progress', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. 评估奖励
        ax2 = plt.subplot(n_envs, 4, env_idx * 4 + 2)
        for algo_name, history in env_results.items():
            if len(history['eval_rewards']) > 0:
                ax2.plot(history['eval_rewards'], 
                        marker='o', 
                        linewidth=linewidths.get(algo_name, 2.0),
                        markersize=4,
                        label=algo_name,
                        color=colors.get(algo_name, 'black'), 
                        linestyle=linestyles.get(algo_name, '-'),
                        alpha=0.9)
        
        ax2.set_xlabel('Evaluation Step', fontsize=12)
        ax2.set_ylabel('Average Reward', fontsize=12)
        ax2.set_title(f'{env_name} - Evaluation Performance', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=9, loc='best')
        ax2.grid(True, alpha=0.3)
        
        # 3. 100回合滚动平均（稳定性）
        ax3 = plt.subplot(n_envs, 4, env_idx * 4 + 3)
        for algo_name, history in env_results.items():
            rewards = history['episode_rewards']
            if len(rewards) >= 100:
                rolling_mean = []
                for i in range(len(rewards) - 99):
                    rolling_mean.append(np.mean(rewards[i:i+100]))
                ax3.plot(range(99, len(rewards)), rolling_mean, 
                        linewidth=linewidths.get(algo_name, 2.0),
                        label=algo_name, 
                        color=colors.get(algo_name, 'black'),
                        linestyle=linestyles.get(algo_name, '-'),
                        alpha=0.9)
        
        ax3.set_xlabel('Episode', fontsize=12)
        ax3.set_ylabel('100-Episode Moving Avg', fontsize=12)
        ax3.set_title(f'{env_name} - Stability Analysis', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=9, loc='best')
        ax3.grid(True, alpha=0.3)
        
        # 4. 最终性能对比（柱状图）
        ax4 = plt.subplot(n_envs, 4, env_idx * 4 + 4)
        final_perfs = {}
        final_stds = {}
        for algo_name, history in env_results.items():
            rewards = history['episode_rewards']
            if len(rewards) > 0:
                last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
                final_perfs[algo_name] = np.mean(last_100)
                final_stds[algo_name] = np.std(last_100)
        
        if final_perfs:
            algos = list(final_perfs.keys())
            perfs = [final_perfs[a] for a in algos]
            stds = [final_stds[a] for a in algos]
            bar_colors = [colors.get(a, 'gray') for a in algos]
            
            x_pos = np.arange(len(algos))
            bars = ax4.bar(x_pos, perfs, yerr=stds, 
                          color=bar_colors, alpha=0.8, capsize=5)
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(algos, rotation=45, ha='right', fontsize=9)
            ax4.set_ylabel('Final Performance (Last 100 Ep)', fontsize=11)
            ax4.set_title(f'{env_name} - Final Performance Comparison', 
                         fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # 在柱状图上显示数值
            for i, (bar, perf) in enumerate(zip(bars, perfs)):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{perf:.1f}',
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_path = f'{save_dir}/ablation_study_results.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] 消融实验图已保存: {save_path}")


def plot_component_contribution(results, save_dir='./ablation_test'):
    """绘制各组件贡献度分析图"""
    fig, axes = plt.subplots(1, len(results), figsize=(7*len(results), 6))
    if len(results) == 1:
        axes = [axes]
    
    for ax, (env_name, env_results) in zip(axes, results.items()):
        if not env_results:
            continue
        
        # 计算各变体相对于标准PPO的提升
        if 'Standard PPO' not in env_results:
            continue
        
        baseline_rewards = env_results['Standard PPO']['episode_rewards']
        if len(baseline_rewards) < 100:
            continue
        baseline_perf = np.mean(baseline_rewards[-100:])
        
        contributions = {}
        for algo_name, history in env_results.items():
            if algo_name == 'Standard PPO':
                continue
            rewards = history['episode_rewards']
            if len(rewards) >= 100:
                perf = np.mean(rewards[-100:])
                improvement = ((perf - baseline_perf) / abs(baseline_perf)) * 100
                contributions[algo_name] = improvement
        
        # 绘制贡献度
        if contributions:
            algos = list(contributions.keys())
            values = [contributions[a] for a in algos]
            colors_list = ['#2E7D32' if v > 0 else '#D32F2F' for v in values]
            
            bars = ax.barh(algos, values, color=colors_list, alpha=0.8)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            ax.set_xlabel('Improvement over Standard PPO (%)', fontsize=12)
            ax.set_title(f'{env_name}\nComponent Contribution Analysis', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # 显示数值
            for bar, value in zip(bars, values):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{value:+.1f}%',
                       ha='left' if width > 0 else 'right',
                       va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_path = f'{save_dir}/component_contribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 组件贡献图已保存: {save_path}")


def print_ablation_analysis(results):
    """打印消融实验分析"""
    print("\n" + "="*80)
    print("消融实验分析报告")
    print("="*80)
    
    for env_name, env_results in results.items():
        if not env_results or len(env_results) == 0:
            continue
        
        print(f"\n{'='*70}")
        print(f"环境: {env_name}")
        print(f"{'='*70}")
        
        # 收集统计数据
        stats = {}
        for algo_name, history in env_results.items():
            rewards = history['episode_rewards']
            if len(rewards) == 0:
                continue
            
            last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
            final_mean = np.mean(last_100)
            final_std = np.std(last_100)
            best_reward = np.max(rewards)
            best_eval = max(history['eval_rewards']) if len(history['eval_rewards']) > 0 else float('-inf')
            
            stats[algo_name] = {
                'final_mean': final_mean,
                'final_std': final_std,
                'best_reward': best_reward,
                'best_eval': best_eval,
                'total_episodes': len(rewards)
            }
        
        # 按最终性能排序
        sorted_algos = sorted(stats.items(), key=lambda x: -x[1]['final_mean'])
        
        print("\n  [性能排名]")
        for rank, (algo_name, stat) in enumerate(sorted_algos, 1):
            print(f"    {rank}. {algo_name}")
            print(f"       最后100回合: {stat['final_mean']:.2f} ± {stat['final_std']:.2f}")
            print(f"       最佳单回合: {stat['best_reward']:.2f}")
            print(f"       最佳评估: {stat['best_eval']:.2f}")
        
        # 组件贡献分析
        if 'Standard PPO' in stats:
            baseline = stats['Standard PPO']['final_mean']
            print(f"\n  [组件贡献分析] (相对于标准PPO: {baseline:.2f})")
            
            component_effects = {
                'Standard Dual PPO': '原始双策略（硬更新 + 原始KL系数）',
                'Improved PPO': '改进版（软更新 + KL系数加底）',
                'No Curiosity': '无好奇心模块',
                'No KL Constraint': '无KL散度约束',
                'No Opponent Policy': '无对手策略（单策略）',
                'No Soft Update': '无软更新（使用硬更新）',
            }
            
            for algo_name, description in component_effects.items():
                if algo_name in stats:
                    perf = stats[algo_name]['final_mean']
                    improvement = ((perf - baseline) / abs(baseline)) * 100
                    symbol = '+' if improvement > 0 else ''
                    print(f"    {algo_name}:")
                    print(f"      组件: {description}")
                    print(f"      性能: {perf:.2f} ({symbol}{improvement:.1f}%)")
        
        # 组件重要性推断
        print(f"\n  [改进效果分析]")
        if 'Standard Dual PPO' in stats and 'Improved PPO' in stats:
            standard_perf = stats['Standard Dual PPO']['final_mean']
            improved_perf = stats['Improved PPO']['final_mean']
            improvement = improved_perf - standard_perf
            improvement_pct = (improvement / abs(standard_perf)) * 100
            print(f"    标准版性能: {standard_perf:.2f}")
            print(f"    改进版性能: {improved_perf:.2f}")
            print(f"    改进幅度: {improvement:+.2f} ({improvement_pct:+.1f}%)")
        
        print(f"\n  [组件重要性推断]")
        required_keys = ['Improved PPO', 'No Curiosity', 'No KL Constraint', 'No Opponent Policy', 'No Soft Update']
        if all(k in stats for k in required_keys):
            improved_perf = stats['Improved PPO']['final_mean']
            no_curiosity = stats['No Curiosity']['final_mean']
            no_kl = stats['No KL Constraint']['final_mean']
            no_opponent = stats['No Opponent Policy']['final_mean']
            no_soft = stats['No Soft Update']['final_mean']
            
            # 估算各组件的边际贡献
            curiosity_contrib = improved_perf - no_curiosity
            kl_contrib = improved_perf - no_kl
            opponent_contrib = improved_perf - no_opponent
            soft_update_contrib = improved_perf - no_soft
            
            print(f"    好奇心模块贡献: {curiosity_contrib:+.2f}")
            print(f"    KL约束贡献: {kl_contrib:+.2f}")
            print(f"    对手策略贡献: {opponent_contrib:+.2f}")
            print(f"    软更新贡献: {soft_update_contrib:+.2f}")
            
            # 找出最重要的组件
            contributions = {
                '好奇心模块': abs(curiosity_contrib),
                'KL约束': abs(kl_contrib),
                '对手策略': abs(opponent_contrib),
                '软更新机制': abs(soft_update_contrib)
            }
            most_important = max(contributions.items(), key=lambda x: x[1])
            print(f"\n    最关键组件: {most_important[0]} (影响度: {most_important[1]:.2f})")


def main():
    print("\n" + "="*80)
    print("消融实验 (Ablation Study) - 双策略PPO组件贡献分析")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 测试环境 - 只测试 MountainCar
    test_envs = {
        'MountainCar-v0': {
            'max_episodes': 1200,     # 增加到 2000 回合（足够学习）
            'update_frequency': 2048,
            'eval_frequency': 50,     # 每 50 回合评估一次
            'log_frequency': 50,      # 每 50 回合输出日志
        },
    }
    
    # 基础配置 - 针对 MountainCar 优化
    base_config = {
        'lr': 3e-4,
        'gamma': 0.99,
        'entropy_coef': 0.01,
        'hidden_dim': 256,
    }
    
    # 消融实验配置
    algorithms = {
        'Standard Dual PPO': {
            'class': DualPolicyPPO,  # 原始版本
            **base_config,
            'intrinsic_coef': 0.02,  # 降低内在奖励系数（从 0.1 → 0.02）
            'kl_coef': 0.005,        # 降低 KL 约束（从 0.01 → 0.005）
        },
        'Improved PPO': {
            'class': ImprovedDualPPO,  # 改进版（完整）
            **base_config,
            'intrinsic_coef': 0.02,  # 降低内在奖励系数
            'kl_coef': 0.005,        # 降低 KL 约束
        },
        'No Curiosity': {
            'class': NoCuriosityPPO,
            **base_config,
            'intrinsic_coef': 0.0,
            'kl_coef': 0.005,
        },
        'No KL Constraint': {
            'class': NoKLConstraintPPO,
            **base_config,
            'intrinsic_coef': 0.02,
            'kl_coef': 0.0,
        },
        'No Opponent Policy': {
            'class': NoOpponentPPO,
            **base_config,
            'intrinsic_coef': 0.02,
            'kl_coef': 0.005,
        },
        'No Soft Update': {
            'class': NoSoftUpdatePPO,
            **base_config,
            'intrinsic_coef': 0.02,
            'kl_coef': 0.005,
        },
        'Standard PPO': {
            'class': StandardPPO,
            **base_config,
            'intrinsic_coef': 0.0,
            'kl_coef': 0.0,
        },
    }
    
    # 创建结果目录
    os.makedirs('./ablation_test', exist_ok=True)
    
    # 存储结果
    all_results = {}
    
    # 运行测试
    for env_name, env_config in test_envs.items():
        print(f"\n\n{'#'*80}")
        print(f"# 环境: {env_name}")
        print(f"{'#'*80}")
        
        all_results[env_name] = {}
        
        for algo_name, algo_config in algorithms.items():
            try:
                config = {**env_config, **algo_config}
                history = run_experiment(
                    env_name,
                    algo_config['class'],
                    algo_name,
                    config,
                    seed=42
                )
                all_results[env_name][algo_name] = history
                
                # 简要统计
                rewards = history['episode_rewards']
                last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
                print(f"\n[OK] {algo_name} 完成")
                print(f"  最后100回合平均: {np.mean(last_100):.2f} ± {np.std(last_100):.2f}")
                if len(history['eval_rewards']) > 0:
                    print(f"  最佳评估: {max(history['eval_rewards']):.2f}")
                
            except Exception as e:
                print(f"\n[ERROR] {algo_name} 失败: {e}")
                import traceback
                traceback.print_exc()
    
    # 绘制结果图
    if any(all_results.values()):
        print("\n" + "="*80)
        print("生成可视化结果")
        print("="*80)
        plot_ablation_results(all_results)
        plot_component_contribution(all_results)
    
    # 打印分析报告
    print_ablation_analysis(all_results)
    
    print("\n" + "="*80)
    print("消融实验完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print(f"\n结果保存在: ./ablation_test/")
    
    # 打印实验说明
    print("\n" + "="*80)
    print("实验说明")
    print("="*80)
    print("""
消融实验设计：

1. Standard Dual PPO (原始版)
   - 好奇心模块 ✓
   - KL散度约束 ✓ (原始公式: 1 - KL)
   - 对手策略 ✓ (硬更新)
   
2. Improved PPO (改进版 - 完整)
   - 好奇心模块 ✓
   - KL散度约束 ✓ (改进公式: 0.1 + 0.9 * (1 - KL))
   - 对手策略 ✓ (软更新 tau=0.01)
   → 完整的改进版本
   
3. No Curiosity (无好奇心)
   - 好奇心模块 ✗
   - 其他改进 ✓
   → 测试好奇心模块的贡献
   
4. No KL Constraint (无KL约束)
   - 好奇心模块 ✓
   - KL散度约束 ✗
   - 其他改进 ✓
   → 测试KL约束的作用
   
5. No Opponent Policy (无对手策略)
   - 好奇心模块 ✓
   - 对手策略 ✗
   - 其他改进 ✓
   → 测试双策略机制的作用
   
6. No Soft Update (无软更新)
   - 所有组件 ✓
   - 软更新 ✗ (使用硬更新)
   → 测试软更新机制的贡献
   
7. Standard PPO (标准PPO)
   - 所有特殊组件 ✗
   → 作为基准对比

通过对比分析，可以量化原始版本、改进版本以及各个改进组件的独立贡献。
    """)


if __name__ == '__main__':
    main()
