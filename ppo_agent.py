import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random

# --- Mạng Neural Actor-Critic ---
class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim_pc, action_dim_pm):
        super(ActorCriticNet, self).__init__()
        # Lớp chung xử lý đặc trưng trạng thái (F)
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        # Nhánh Actor: Tính xác suất cho Pc và Pm (mỗi loại 10 mức)
        self.actor_pc = nn.Linear(64, action_dim_pc)
        self.actor_pm = nn.Linear(64, action_dim_pm)
        # Nhánh Critic: Đánh giá giá trị trạng thái V(s)
        self.critic = nn.Linear(64, 1)

    def forward(self, state):
        features = self.backbone(state)
        # Trả về phân phối xác suất (Softmax)
        probs_pc = torch.softmax(self.actor_pc(features), dim=-1)
        probs_pm = torch.softmax(self.actor_pm(features), dim=-1)
        # Trả về giá trị trạng thái
        state_value = self.critic(features)
        return probs_pc, probs_pm, state_value

# --- Lớp PPO Agent chính ---
class PPOAgent:
    def __init__(self, max_generations=200):
        self.max_generations = max_generations
        
        # Không gian hành động giống file gốc của bạn
        self.pc_actions = [0.4 + i * 0.05 for i in range(10)]
        self.pm_actions = [0.01 + i * 0.02 for i in range(10)]
        self.initial_metrics = {'A1': None, 'D1': None, 'B1': None}

        # Thiết lập mạng Neural
        self.state_dim = 1  # Biến F
        self.policy = ActorCriticNet(self.state_dim, 10, 10)
        self.policy_old = ActorCriticNet(self.state_dim, 10, 10)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.002)
        self.MseLoss = nn.MSELoss()
        
        # Hyperparameters PPO
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 4 # Số lần cập nhật lại trên cùng một tập dữ liệu
        self.memory = []

    def calculate_metrics(self, population):
        # Giữ nguyên logic tính toán của bạn
        values = [ind.makespan for ind in population]
        N = len(population)
        A_t = sum(values) / N
        B_t = min(values)
        variance = sum((x - A_t) ** 2 for x in values) / N
        D_t = np.sqrt(variance)
        return A_t, D_t, B_t

    def get_state(self, population, generation):
        """Trả về trạng thái F dưới dạng Tensor thay vì ép kiểu int"""
        A_t, D_t, B_t = self.calculate_metrics(population)
        if self.initial_metrics['A1'] is None or generation <= 1:
            self.initial_metrics = {'A1': A_t or 1.0, 'D1': D_t or 1.0, 'B1': B_t or 1.0}
            return torch.FloatTensor([1.0])
        
        w1, w2, w3 = 0.35, 0.35, 0.3
        F = w1*(A_t/self.initial_metrics['A1']) + w2*(D_t/self.initial_metrics['D1']) + w3*(B_t/self.initial_metrics['B1'])
        return torch.FloatTensor([F])

    def select_action(self, state):
        """Chọn hành động dựa trên xác suất từ mạng Neural"""
        with torch.no_grad():
            probs_pc, probs_pm, _ = self.policy_old(state)
        
        # Tạo phân phối xác suất
        dist_pc = Categorical(probs_pc)
        dist_pm = Categorical(probs_pm)
        
        # Lấy mẫu hành động (Sampling)
        idx_pc = dist_pc.sample()
        idx_pm = dist_pm.sample()
        
        # Lưu thông tin bước này vào memory
        self.current_step_info = {
            'state': state,
            'idx_pc': idx_pc,
            'idx_pm': idx_pm,
            'log_prob_pc': dist_pc.log_prob(idx_pc),
            'log_prob_pm': dist_pm.log_prob(idx_pm)
        }
        
        # Trả về giá trị thực tế để EA sử dụng
        val_pc = min(1.0, self.pc_actions[idx_pc.item()] + random.uniform(0, 0.05))
        val_pm = min(1.0, self.pm_actions[idx_pm.item()] + random.uniform(0, 0.02))
        
        return val_pc, val_pm

    def calculate_reward(self, population):
        # Giữ nguyên logic Reward: Tốt = 1, Tệ = -1
        A_t, D_t, _ = self.calculate_metrics(population)
        R = 0.5 * (A_t / self.initial_metrics['A1']) + 0.5 * (D_t / self.initial_metrics['D1'])
        return 1.0 if R <= 1.0 else -1.0

    def update_policy(self, next_population):
        """Lưu reward và tiến hành cập nhật nếu đủ batch"""
        reward = self.calculate_reward(next_population)
        self.current_step_info['reward'] = reward
        self.memory.append(self.current_step_info)
        
        # Cập nhật sau mỗi 10 thế hệ (Batch size = 10)
        if len(self.memory) >= 10:
            self._ppo_update()
            self.memory = []

def _ppo_update(self):
        # 1. Chuẩn bị dữ liệu
        old_states = torch.stack([m['state'] for m in self.memory]).detach()
        old_log_pc = torch.stack([m['log_prob_pc'] for m in self.memory]).detach()
        old_log_pm = torch.stack([m['log_prob_pm'] for m in self.memory]).detach()
        rewards = torch.tensor([m['reward'] for m in self.memory], dtype=torch.float)
        old_idx_pc = torch.stack([m['idx_pc'] for m in self.memory]).detach()
        old_idx_pm = torch.stack([m['idx_pm'] for m in self.memory]).detach()

        # 2. Tối ưu hóa: Chuẩn hóa Reward/Advantage giúp ổn định hơn
        # (Chuyển reward về phân phối chuẩn có mean=0, std=1)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        for _ in range(self.K_epochs):
            probs_pc, probs_pm, state_values = self.policy(old_states)
            state_values = state_values.squeeze() # Đưa về dạng [Batch_Size]
            
            dist_pc = Categorical(probs_pc)
            dist_pm = Categorical(probs_pm)
            
            # Tính Ratio
            ratio_pc = torch.exp(dist_pc.log_prob(old_idx_pc) - old_log_pc)
            ratio_pm = torch.exp(dist_pm.log_prob(old_idx_pm) - old_log_pm)
            
            # Tính Advantage
            advantages = rewards - state_values.detach()
            
            # PPO Loss Clip cho Pc
            surr1_pc = ratio_pc * advantages
            surr2_pc = torch.clamp(ratio_pc, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss_pc = -torch.min(surr1_pc, surr2_pc).mean()

            # PPO Loss Clip cho Pm
            surr1_pm = ratio_pm * advantages
            surr2_pm = torch.clamp(ratio_pm, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss_pm = -torch.min(surr1_pm, surr2_pm).mean()
            
            # Critic Loss (MSE)
            loss_v = self.MseLoss(state_values, rewards)
            
            # Entropy Loss (Khuyến khích khám phá - Tùy chọn nhưng nên có)
            loss_entropy = -(dist_pc.entropy().mean() + dist_pm.entropy().mean())
            
            # Tổng Loss (0.01 là trọng số cho Entropy)
            total_loss = loss_pc + loss_pm + 0.5 * loss_v + 0.01 * loss_entropy
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())