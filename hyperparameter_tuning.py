"""
超参数调优实验：针对 Original Dual PPO 和 Improved Dual PPO 进行大范围超参数探索

改进点：
1. 为两种dual PPO方法设计超参数网格
2. Standard PPO 只运行一次作为基准
3. 实时保存中间结果，防止实验中断导致数据丢失
4. 图片使用不同名称，避免覆盖
5. 详细的调试信息输出
"""

import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
import os
import json
import time
from datetime import datetime
from dual_policy_ppo import DualPolicyPPO
from trainer import Trainer
import itertools

# ==================== 定义算法类 ====================

class StandardPPO(DualPolicyPPO):
    """标准PPO（无好奇心，无双策略）"""
    
    def compute_intrinsic_reward(self, states, actions, next_states):
        return torch.zeros(len(states), device=self.device)
    
    def update(self, memory, batch_size=64, n_epochs=10):
        return super().update(memory, batch_size, n_epochs)


class ImprovedDualPPO(DualPolicyPPO):
    """
    改进版 Dual PPO
    1. 软更新 Opponent
    2. 调整 KL 惩罚系数
    """
    def __init__(self, *args, tau=0.01, kl_base=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau  # 软更新系数
        self.kl_base = kl_base  # KL系数底数

    def compute_intrinsic_reward(self, states, actions, next_states):
        """
        计算改进的内在奖励
        r_intrinsic = Curiosity(s, a) * [kl_base + (1-kl_base) * (1 - KL_penalty)]
        """
        # 好奇心奖励
        curiosity_reward, _, _ = self.curiosity(states, actions, next_states)
        
        # KL散度（归一化到[0, 1]）
        kl_div = self.compute_kl_divergence(states)
        kl_penalty = torch.clamp(kl_div / self.kl_threshold, 0, 1)
        
        # 修改点：给系数加底
        coefficient = self.kl_base + (1 - self.kl_base) * (1 - kl_penalty)
        
        intrinsic_reward = curiosity_reward * coefficient
        
        return intrinsic_reward.detach()

    def update(self, memory, batch_size=64, n_epochs=10):
        # 执行父类更新 (梯度下降)
        stats = super().update(memory, batch_size, n_epochs)
        
        # 修改点：软更新 Opponent
        for param, target_param in zip(self.actor_critic.parameters(), self.opponent_actor_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return stats

    def update_opponent_policy(self, current_performance):
        # 禁用父类的硬更新逻辑
        self.performance_history.append(current_performance)
        return False


# ==================== 实验运行函数 ====================

def run_single_experiment(env_name, agent_class, config_name, config, seed=42, results_dir='./hyperparameter_tuning'):
    """运行单个实验配置"""
    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始测试: {config_name}")
    print(f"环境: {env_name}")
    print(f"配置: {config}")
    print(f"{'='*70}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        continuous = False
    else:
        action_dim = env.action_space.shape[0]
        continuous = True
    
    env.close()
    
    agent_params = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'continuous': continuous,
        'lr': config['lr'],
        'gamma': config['gamma'],
        'intrinsic_coef': config.get('intrinsic_coef', 0.0),
        'kl_coef': config.get('kl_coef', 0.0),
        'entropy_coef': config['entropy_coef'],
        'hidden_dim': config['hidden_dim'],
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # 添加改进版专有参数
    if agent_class == ImprovedDualPPO:
        agent_params['tau'] = config.get('tau', 0.01)
        agent_params['kl_base'] = config.get('kl_base', 0.1)
    
    agent = agent_class(**agent_params)
    
    # 保存到 hyperparameter_tuning 目录
    safe_name = config_name.replace(' ', '_').replace('(', '').replace(')', '').replace(':', '').replace('=', '')
    save_dir = f'{results_dir}/{env_name}_{safe_name}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存配置到文件
    config_file = f'{save_dir}/config.json'
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 配置已保存: {config_file}")
    
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
    
    start_time = time.time()
    history = trainer.train()
    elapsed_time = time.time() - start_time
    
    trainer.close()
    
    # 保存训练历史
    history_file = f'{save_dir}/history.json'
    serializable_history = {
        'episode_rewards': [float(x) for x in history['episode_rewards']],
        'eval_rewards': [float(x) for x in history['eval_rewards']],
        'elapsed_time': elapsed_time
    }
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_history, f, indent=2)
    print(f"[INFO] 训练历史已保存: {history_file}")
    
    # 计算最终性能指标
    final_100_avg = np.mean(history['episode_rewards'][-100:]) if len(history['episode_rewards']) >= 100 else np.mean(history['episode_rewards'])
    best_eval = max(history['eval_rewards']) if history['eval_rewards'] else -np.inf
    
    print(f"[完成] {config_name}")
    print(f"  - 最后100回合平均: {final_100_avg:.2f}")
    print(f"  - 最佳评估奖励: {best_eval:.2f}")
    print(f"  - 训练用时: {elapsed_time/60:.2f} 分钟")
    
    return history, final_100_avg, best_eval


# ==================== 超参数配置 ====================

def generate_hyperparameter_configs():
    """生成超参数配置空间"""
    
    # Original Dual PPO 超参数网格
    original_dual_grid = {
        'lr': [1e-4, 3e-4, 5e-4],
        'intrinsic_coef': [0.01, 0.02, 0.05, 0.1],
        'kl_coef': [0.0005, 0.001, 0.002],
        'entropy_coef': [0.005, 0.01, 0.02],
        'gamma': [0.99, 0.995]
    }
    
    # Improved Dual PPO 超参数网格
    improved_dual_grid = {
        'lr': [1e-4, 3e-4, 5e-4],
        'intrinsic_coef': [0.01, 0.02, 0.05, 0.1],
        'kl_coef': [0.0005, 0.001, 0.002],
        'entropy_coef': [0.005, 0.01, 0.02],
        'tau': [0.005, 0.01, 0.02],
        'kl_base': [0.05, 0.1, 0.15],
        'gamma': [0.99, 0.995]
    }
    
    return original_dual_grid, improved_dual_grid


def sample_random_configs(param_grid, n_samples=10):
    """从参数网格中随机采样配置"""
    configs = []
    keys = list(param_grid.keys())
    
    for _ in range(n_samples):
        config = {}
        for key in keys:
            config[key] = np.random.choice(param_grid[key])
        configs.append(config)
    
    return configs


# ==================== 绘图函数 ====================

def plot_tuning_results(all_results, save_path='./hyperparameter_tuning/tuning_results.png'):
    """绘制超参数调优结果对比"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # 准备数据
    standard_ppo_results = all_results.get('Standard PPO', {})
    original_dual_results = all_results.get('Original Dual PPO', [])
    improved_dual_results = all_results.get('Improved Dual PPO', [])
    
    # 1. 训练曲线对比（左上）
    ax1 = plt.subplot(2, 3, 1)
    
    # Standard PPO
    if standard_ppo_results:
        history = standard_ppo_results['history']
        rewards = history['episode_rewards']
        if len(rewards) > 0:
            window = min(50, len(rewards) // 10) if len(rewards) > 10 else 1
            if window > 1:
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                x = range(window-1, len(rewards))
                ax1.plot(x, smoothed, linewidth=2.5, label='Standard PPO', 
                        color='gray', linestyle='--', alpha=0.9)
    
    # Original Dual PPO - 显示前5个最好的配置
    if original_dual_results:
        sorted_original = sorted(original_dual_results, key=lambda x: x['final_100_avg'], reverse=True)[:5]
        for i, result in enumerate(sorted_original):
            history = result['history']
            rewards = history['episode_rewards']
            if len(rewards) > 0:
                window = min(50, len(rewards) // 10) if len(rewards) > 10 else 1
                if window > 1:
                    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    x = range(window-1, len(rewards))
                    ax1.plot(x, smoothed, linewidth=1.5, 
                            label=f'Original #{i+1}', 
                            color='blue', alpha=0.3 + i*0.1)
    
    # Improved Dual PPO - 显示前5个最好的配置
    if improved_dual_results:
        sorted_improved = sorted(improved_dual_results, key=lambda x: x['final_100_avg'], reverse=True)[:5]
        for i, result in enumerate(sorted_improved):
            history = result['history']
            rewards = history['episode_rewards']
            if len(rewards) > 0:
                window = min(50, len(rewards) // 10) if len(rewards) > 10 else 1
                if window > 1:
                    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    x = range(window-1, len(rewards))
                    ax1.plot(x, smoothed, linewidth=1.5, 
                            label=f'Improved #{i+1}', 
                            color='red', alpha=0.3 + i*0.1)
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward (Smoothed)', fontsize=12)
    ax1.set_title('Training Progress - Top 5 Configs', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. 最终性能分布（右上）
    ax2 = plt.subplot(2, 3, 2)
    
    performance_data = []
    labels = []
    
    if standard_ppo_results:
        performance_data.append([standard_ppo_results['final_100_avg']])
        labels.append('Standard\nPPO')
    
    if original_dual_results:
        original_perfs = [r['final_100_avg'] for r in original_dual_results]
        performance_data.append(original_perfs)
        labels.append(f'Original\nDual PPO\n(n={len(original_perfs)})')
    
    if improved_dual_results:
        improved_perfs = [r['final_100_avg'] for r in improved_dual_results]
        performance_data.append(improved_perfs)
        labels.append(f'Improved\nDual PPO\n(n={len(improved_perfs)})')
    
    bp = ax2.boxplot(performance_data, labels=labels, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     widths=0.6)
    
    ax2.set_ylabel('Final 100-Episode Average Reward', fontsize=12)
    ax2.set_title('Performance Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 最佳评估奖励对比（中上）
    ax3 = plt.subplot(2, 3, 3)
    
    best_eval_data = []
    
    if standard_ppo_results:
        best_eval_data.append([standard_ppo_results['best_eval']])
    
    if original_dual_results:
        original_evals = [r['best_eval'] for r in original_dual_results]
        best_eval_data.append(original_evals)
    
    if improved_dual_results:
        improved_evals = [r['best_eval'] for r in improved_dual_results]
        best_eval_data.append(improved_evals)
    
    bp2 = ax3.boxplot(best_eval_data, labels=labels, patch_artist=True,
                      boxprops=dict(facecolor='lightgreen', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2),
                      widths=0.6)
    
    ax3.set_ylabel('Best Evaluation Reward', fontsize=12)
    ax3.set_title('Evaluation Performance', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Original Dual PPO 超参数影响分析（左下）
    ax4 = plt.subplot(2, 3, 4)
    if original_dual_results and len(original_dual_results) > 0:
        # 分析 intrinsic_coef 的影响
        ic_values = {}
        for result in original_dual_results:
            ic = result['config']['intrinsic_coef']
            if ic not in ic_values:
                ic_values[ic] = []
            ic_values[ic].append(result['final_100_avg'])
        
        ics = sorted(ic_values.keys())
        means = [np.mean(ic_values[ic]) for ic in ics]
        stds = [np.std(ic_values[ic]) for ic in ics]
        
        ax4.errorbar(ics, means, yerr=stds, marker='o', linewidth=2, 
                    markersize=8, capsize=5, color='blue')
        ax4.set_xlabel('Intrinsic Coefficient', fontsize=12)
        ax4.set_ylabel('Avg Performance', fontsize=12)
        ax4.set_title('Original Dual PPO: Intrinsic Coef Impact', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    
    # 5. Improved Dual PPO tau参数影响（中下）
    ax5 = plt.subplot(2, 3, 5)
    if improved_dual_results and len(improved_dual_results) > 0:
        # 分析 tau 的影响
        tau_values = {}
        for result in improved_dual_results:
            tau = result['config']['tau']
            if tau not in tau_values:
                tau_values[tau] = []
            tau_values[tau].append(result['final_100_avg'])
        
        taus = sorted(tau_values.keys())
        means = [np.mean(tau_values[tau]) for tau in taus]
        stds = [np.std(tau_values[tau]) for tau in taus]
        
        ax5.errorbar(taus, means, yerr=stds, marker='s', linewidth=2, 
                    markersize=8, capsize=5, color='red')
        ax5.set_xlabel('Tau (Soft Update Rate)', fontsize=12)
        ax5.set_ylabel('Avg Performance', fontsize=12)
        ax5.set_title('Improved Dual PPO: Tau Impact', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
    
    # 6. 学习率影响对比（右下）
    ax6 = plt.subplot(2, 3, 6)
    
    # Original Dual PPO
    if original_dual_results:
        lr_values = {}
        for result in original_dual_results:
            lr = result['config']['lr']
            if lr not in lr_values:
                lr_values[lr] = []
            lr_values[lr].append(result['final_100_avg'])
        
        lrs = sorted(lr_values.keys())
        means = [np.mean(lr_values[lr]) for lr in lrs]
        ax6.plot(lrs, means, marker='o', linewidth=2, markersize=8, 
                label='Original Dual', color='blue')
    
    # Improved Dual PPO
    if improved_dual_results:
        lr_values = {}
        for result in improved_dual_results:
            lr = result['config']['lr']
            if lr not in lr_values:
                lr_values[lr] = []
            lr_values[lr].append(result['final_100_avg'])
        
        lrs = sorted(lr_values.keys())
        means = [np.mean(lr_values[lr]) for lr in lrs]
        ax6.plot(lrs, means, marker='s', linewidth=2, markersize=8, 
                label='Improved Dual', color='red')
    
    ax6.set_xlabel('Learning Rate', fontsize=12)
    ax6.set_ylabel('Avg Performance', fontsize=12)
    ax6.set_title('Learning Rate Impact Comparison', fontsize=12, fontweight='bold')
    ax6.set_xscale('log')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] 调优结果图已保存: {save_path}")
    plt.close()


def save_summary_report(all_results, save_path='./hyperparameter_tuning/summary_report.txt'):
    """保存文本总结报告"""
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("超参数调优实验总结报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Standard PPO
        if 'Standard PPO' in all_results:
            f.write("【Standard PPO (Baseline)】\n")
            result = all_results['Standard PPO']
            f.write(f"  最后100回合平均: {result['final_100_avg']:.2f}\n")
            f.write(f"  最佳评估奖励: {result['best_eval']:.2f}\n")
            f.write(f"  配置: {result['config']}\n\n")
        
        # Original Dual PPO
        if 'Original Dual PPO' in all_results and all_results['Original Dual PPO']:
            f.write("【Original Dual PPO】\n")
            results = all_results['Original Dual PPO']
            f.write(f"  测试配置数: {len(results)}\n")
            
            sorted_results = sorted(results, key=lambda x: x['final_100_avg'], reverse=True)
            
            f.write(f"  最佳性能: {sorted_results[0]['final_100_avg']:.2f}\n")
            f.write(f"  平均性能: {np.mean([r['final_100_avg'] for r in results]):.2f}\n")
            f.write(f"  性能标准差: {np.std([r['final_100_avg'] for r in results]):.2f}\n\n")
            
            f.write("  Top 3 配置:\n")
            for i, result in enumerate(sorted_results[:3]):
                f.write(f"    #{i+1} (平均奖励: {result['final_100_avg']:.2f})\n")
                f.write(f"      {result['config']}\n")
            f.write("\n")
        
        # Improved Dual PPO
        if 'Improved Dual PPO' in all_results and all_results['Improved Dual PPO']:
            f.write("【Improved Dual PPO (Soft Update)】\n")
            results = all_results['Improved Dual PPO']
            f.write(f"  测试配置数: {len(results)}\n")
            
            sorted_results = sorted(results, key=lambda x: x['final_100_avg'], reverse=True)
            
            f.write(f"  最佳性能: {sorted_results[0]['final_100_avg']:.2f}\n")
            f.write(f"  平均性能: {np.mean([r['final_100_avg'] for r in results]):.2f}\n")
            f.write(f"  性能标准差: {np.std([r['final_100_avg'] for r in results]):.2f}\n\n")
            
            f.write("  Top 3 配置:\n")
            for i, result in enumerate(sorted_results[:3]):
                f.write(f"    #{i+1} (平均奖励: {result['final_100_avg']:.2f})\n")
                f.write(f"      {result['config']}\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
    
    print(f"[OK] 总结报告已保存: {save_path}")


# ==================== 主函数 ====================

def main():
    print("\n" + "="*80)
    print("超参数调优实验")
    print("目标: 为 Original Dual PPO 和 Improved Dual PPO 探索最优超参数")
    print("="*80)
    
    # 实验基础配置
    env_name = 'MountainCar-v0'
    env_config = {
        'max_episodes': 600,
        'update_frequency': 2048,
        'eval_frequency': 50,
        'log_frequency': 50,
        'hidden_dim': 256
    }
    
    results_dir = './hyperparameter_tuning'
    os.makedirs(results_dir, exist_ok=True)
    
    # 存储所有结果
    all_results = {
        'Standard PPO': {},
        'Original Dual PPO': [],
        'Improved Dual PPO': []
    }
    
    # 生成超参数配置
    original_dual_grid, improved_dual_grid = generate_hyperparameter_configs()
    
    # 采样配置（从网格中随机采样以减少总实验数）
    n_samples_per_method = 12  # 每个方法测试12个配置
    
    print(f"\n[INFO] 为每种方法随机采样 {n_samples_per_method} 个超参数配置")
    
    original_configs = sample_random_configs(original_dual_grid, n_samples_per_method)
    improved_configs = sample_random_configs(improved_dual_grid, n_samples_per_method)
    
    # ========== 1. 运行 Standard PPO (Baseline) ==========
    print("\n" + "="*80)
    print("第1步: 运行 Standard PPO (Baseline)")
    print("="*80)
    
    standard_config = {
        **env_config,
        'lr': 3e-4,
        'gamma': 0.99,
        'intrinsic_coef': 0.0,
        'kl_coef': 0.0,
        'entropy_coef': 0.01
    }
    
    try:
        history, final_100_avg, best_eval = run_single_experiment(
            env_name,
            StandardPPO,
            'Standard PPO',
            standard_config,
            seed=42,
            results_dir=results_dir
        )
        all_results['Standard PPO'] = {
            'history': history,
            'final_100_avg': final_100_avg,
            'best_eval': best_eval,
            'config': standard_config
        }
        
        # 保存中间结果
        with open(f'{results_dir}/intermediate_results.json', 'w', encoding='utf-8') as f:
            json.dump({
                'Standard PPO': {
                    'final_100_avg': float(final_100_avg),
                    'best_eval': float(best_eval),
                    'config': standard_config
                }
            }, f, indent=2)
        print("[INFO] Standard PPO 结果已保存")
        
    except Exception as e:
        print(f"[ERROR] Standard PPO 失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== 2. 运行 Original Dual PPO 配置 ==========
    print("\n" + "="*80)
    print(f"第2步: 测试 Original Dual PPO ({len(original_configs)} 个配置)")
    print("="*80)
    
    for idx, params in enumerate(original_configs):
        config = {**env_config, **params}
        config_name = f"Original Dual PPO Config{idx+1}"
        
        try:
            history, final_100_avg, best_eval = run_single_experiment(
                env_name,
                DualPolicyPPO,
                config_name,
                config,
                seed=42,
                results_dir=results_dir
            )
            
            all_results['Original Dual PPO'].append({
                'history': history,
                'final_100_avg': final_100_avg,
                'best_eval': best_eval,
                'config': params,
                'config_name': config_name
            })
            
            # 实时保存中间结果
            intermediate_file = f'{results_dir}/intermediate_original_dual.json'
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump([{
                    'config': r['config'],
                    'final_100_avg': float(r['final_100_avg']),
                    'best_eval': float(r['best_eval'])
                } for r in all_results['Original Dual PPO']], f, indent=2)
            
            print(f"[INFO] Original Dual PPO 进度: {idx+1}/{len(original_configs)}")
            
        except Exception as e:
            print(f"[ERROR] {config_name} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # ========== 3. 运行 Improved Dual PPO 配置 ==========
    print("\n" + "="*80)
    print(f"第3步: 测试 Improved Dual PPO ({len(improved_configs)} 个配置)")
    print("="*80)
    
    for idx, params in enumerate(improved_configs):
        config = {**env_config, **params}
        config_name = f"Improved Dual PPO Config{idx+1}"
        
        try:
            history, final_100_avg, best_eval = run_single_experiment(
                env_name,
                ImprovedDualPPO,
                config_name,
                config,
                seed=42,
                results_dir=results_dir
            )
            
            all_results['Improved Dual PPO'].append({
                'history': history,
                'final_100_avg': final_100_avg,
                'best_eval': best_eval,
                'config': params,
                'config_name': config_name
            })
            
            # 实时保存中间结果
            intermediate_file = f'{results_dir}/intermediate_improved_dual.json'
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump([{
                    'config': r['config'],
                    'final_100_avg': float(r['final_100_avg']),
                    'best_eval': float(r['best_eval'])
                } for r in all_results['Improved Dual PPO']], f, indent=2)
            
            print(f"[INFO] Improved Dual PPO 进度: {idx+1}/{len(improved_configs)}")
            
        except Exception as e:
            print(f"[ERROR] {config_name} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # ========== 4. 生成对比图表和报告 ==========
    print("\n" + "="*80)
    print("第4步: 生成分析结果")
    print("="*80)
    
    # 绘制对比图
    plot_tuning_results(all_results, save_path=f'{results_dir}/tuning_results.png')
    
    # 保存总结报告
    save_summary_report(all_results, save_path=f'{results_dir}/summary_report.txt')
    
    # 保存完整结果
    final_results_file = f'{results_dir}/all_results.json'
    with open(final_results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'Standard PPO': {
                'config': all_results['Standard PPO'].get('config', {}),
                'final_100_avg': float(all_results['Standard PPO'].get('final_100_avg', 0)),
                'best_eval': float(all_results['Standard PPO'].get('best_eval', 0))
            },
            'Original Dual PPO': [{
                'config': r['config'],
                'final_100_avg': float(r['final_100_avg']),
                'best_eval': float(r['best_eval'])
            } for r in all_results['Original Dual PPO']],
            'Improved Dual PPO': [{
                'config': r['config'],
                'final_100_avg': float(r['final_100_avg']),
                'best_eval': float(r['best_eval'])
            } for r in all_results['Improved Dual PPO']]
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] 完整结果已保存: {final_results_file}")
    
    # 打印最佳配置
    print("\n" + "="*80)
    print("最佳超参数配置")
    print("="*80)
    
    if all_results['Original Dual PPO']:
        best_original = max(all_results['Original Dual PPO'], key=lambda x: x['final_100_avg'])
        print(f"\n【Original Dual PPO 最佳配置】")
        print(f"  性能: {best_original['final_100_avg']:.2f}")
        print(f"  配置: {best_original['config']}")
    
    if all_results['Improved Dual PPO']:
        best_improved = max(all_results['Improved Dual PPO'], key=lambda x: x['final_100_avg'])
        print(f"\n【Improved Dual PPO 最佳配置】")
        print(f"  性能: {best_improved['final_100_avg']:.2f}")
        print(f"  配置: {best_improved['config']}")
    
    print("\n" + "="*80)
    print("实验完成！所有结果保存在:", results_dir)
    print("="*80)


if __name__ == '__main__':
    main()

