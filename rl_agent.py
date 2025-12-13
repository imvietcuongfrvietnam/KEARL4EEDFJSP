import numpy as np
import random

class RLAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon_start=0.9, epsilon_min=0.05, max_generations=200):
        """
        Khởi tạo Agent cho KEARL hỗ trợ cả Q-Learning và SARSA.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.max_generations = max_generations
        
        # --- 1. STATE SPACE (21 trạng thái) ---
        self.num_states = 21 
        self.initial_metrics = {'A1': None, 'D1': None, 'B1': None}
        
        # --- 2. ACTION SPACE (10 hành động mỗi loại) ---
        self.num_actions_pc = 10
        self.pc_actions = [0.4 + i * 0.05 for i in range(10)]
        
        self.num_actions_pm = 10
        self.pm_actions = [0.01 + i * 0.02 for i in range(10)]
        
        # --- 3. Q-TABLES ---
        self.q_table_pc = np.zeros((self.num_states, self.num_actions_pc))
        self.q_table_pm = np.zeros((self.num_states, self.num_actions_pm))
        
        # Lưu trạng thái và hành động của bước t (để update ở bước t+1)
        self.last_state = 0
        self.last_action_idx_pc = 0
        self.last_action_idx_pm = 0

    # ... (Giữ nguyên hàm calculate_metrics và get_state) ...
    def calculate_metrics(self, population):
        values = [ind.makespan for ind in population]
        N = len(population)
        A_t = sum(values) / N
        B_t = min(values)
        variance = sum((x - A_t) ** 2 for x in values) / N
        D_t = np.sqrt(variance)
        return A_t, D_t, B_t

    def get_state(self, population, generation):
        A_t, D_t, B_t = self.calculate_metrics(population)
        
        if self.initial_metrics['A1'] is None or generation == 1:
            self.initial_metrics['A1'] = A_t if A_t > 0 else 1.0
            self.initial_metrics['D1'] = D_t if D_t > 0 else 1.0
            self.initial_metrics['B1'] = B_t if B_t > 0 else 1.0
            self.last_state = 0
            return 0
        
        w1, w2, w3 = 0.35, 0.35, 0.3
        F = w1*(A_t/self.initial_metrics['A1']) + w2*(D_t/self.initial_metrics['D1']) + w3*(B_t/self.initial_metrics['B1'])
        
        state = int(F / 0.05)
        if state >= 20: state = 20
        return state

    def _choose_action_index(self, q_table, state):
        """Helper: Chọn index hành động dựa trên Epsilon-Greedy (cho nội bộ class dùng)"""
        if random.random() > self.epsilon:
            # Exploitation
            # Nếu có nhiều max bằng nhau thì chọn ngẫu nhiên trong số đó (tránh bias)
            max_val = np.max(q_table[state])
            candidates = np.where(q_table[state] == max_val)[0]
            return random.choice(candidates)
        else:
            # Exploration
            return random.randint(0, q_table.shape[1] - 1)

    def select_action(self, state, current_gen):
        """
        Chọn hành động (Pc, Pm) cho thế hệ hiện tại.
        """
        # Cập nhật Epsilon decay (Eq. 34)
        decay = current_gen * ((self.epsilon_start - self.epsilon_min) / self.max_generations)
        self.epsilon = max(self.epsilon_min, self.epsilon_start - decay)
        
        # Chọn index hành động
        idx_pc = self._choose_action_index(self.q_table_pc, state)
        idx_pm = self._choose_action_index(self.q_table_pm, state)
        
        # Lưu lại để update sau
        self.last_state = state
        self.last_action_idx_pc = idx_pc
        self.last_action_idx_pm = idx_pm
        
        # Chuyển đổi sang giá trị thực + nhiễu ngẫu nhiên
        base_pc = self.pc_actions[idx_pc]
        base_pm = self.pm_actions[idx_pm]
        
        final_pc = min(1.0, base_pc + random.uniform(0, 0.05))
        final_pm = min(1.0, base_pm + random.uniform(0, 0.02))
        
        return final_pc, final_pm

    def calculate_reward(self, population):
        """Tính Reward (Eq. 31, 32)"""
        A_t, D_t, _ = self.calculate_metrics(population)
        term1 = 0.5 * (A_t / self.initial_metrics['A1'])
        term2 = 0.5 * (D_t / self.initial_metrics['D1'])
        R = term1 + term2
        
        # Lưu ý: Với bài toán Min (Makespan), R giảm là tốt. 
        # Nếu R > 1 (tức là Fitness tăng -> Tệ đi) -> Reward = -1
        # Nếu R <= 1 (Fitness giảm/giữ -> Tốt lên) -> Reward = 1
        # (Điều chỉnh logic cho phù hợp bài toán Min)
        if R <= 1.0: 
            return 1
        else:
            return -1

    def update_policy(self, next_population, method='q_learning'):
        """
        Cập nhật Q-Table dùng Q-Learning HOẶC SARSA.
        
        Args:
            next_population: Quần thể thế hệ t+1.
            method (str): 'q_learning' hoặc 'sarsa'.
        """
        reward = self.calculate_reward(next_population)
        next_state = self.get_state(next_population, -1)
        
        # 1. Update Q-Table cho Pc
        q_curr_pc = self.q_table_pc[self.last_state, self.last_action_idx_pc]
        
        if method == 'q_learning':
            # Eq. 25: Max Q(s', a')
            target_pc = np.max(self.q_table_pc[next_state])
        else: # SARSA
            # Eq. 26: Q(s', a') với a' được chọn theo chính sách hiện tại
            next_action_pc = self._choose_action_index(self.q_table_pc, next_state)
            target_pc = self.q_table_pc[next_state, next_action_pc]

        new_q_pc = (1 - self.alpha) * q_curr_pc + self.alpha * (reward + self.gamma * target_pc)
        self.q_table_pc[self.last_state, self.last_action_idx_pc] = new_q_pc
        
        # 2. Update Q-Table cho Pm
        q_curr_pm = self.q_table_pm[self.last_state, self.last_action_idx_pm]
        
        if method == 'q_learning':
            target_pm = np.max(self.q_table_pm[next_state])
        else: # SARSA
            next_action_pm = self._choose_action_index(self.q_table_pm, next_state)
            target_pm = self.q_table_pm[next_state, next_action_pm]

        new_q_pm = (1 - self.alpha) * q_curr_pm + self.alpha * (reward + self.gamma * target_pm)
        self.q_table_pm[self.last_state, self.last_action_idx_pm] = new_q_pm