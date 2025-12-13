import random
import numpy as np

# --- 1. PARAMETER CLASS ---
class Parameter:
    """Lưu trữ tham số toàn cục (Giữ nguyên như bạn đã làm tốt)"""
    def __init__(self):
        self.n = 0  # Số Job
        self.m = 0  # Số Máy
        self.AC = 10.0      # Common Energy (Eq. 8)
        self.UT_k = 2.0     # Transport Energy Unit
        self.LI = 1e9       # Big M
        
        # Breakdown params (Eq. 22-24)
        self.lambda_0 = 0.01; self.lambda_1 = 0.5; self.lambda_2 = 0.3; self.lambda_3 = 0.2
        self.alpha_1 = 0.2; self.alpha_2 = 0.3; self.alpha_3 = 0.3; self.alpha_4 = 0.2
        self.beta_0 = 5.0; self.beta_1 = 0.5; self.beta_2 = 0.5; self.gamma = 0.1
        
        # Ma trận thời gian vận chuyển (quan trọng cho Eq. 6)
        self.TT_matrix = [] 

# --- 2. MACHINE CLASS ---
class Machine:
    def __init__(self, machine_id, age=0, energy_idle_unit=1.0):
        self.machine_id = machine_id
        
        # Static Parameters
        self.AI = energy_idle_unit # AI_k: Idle Energy (Eq. 7)
        
        # Dynamic State (Breakdown logic)
        self.v = age            # Age
        self.rho_k = 0          # Repair count
        self.T_k = 0.0          # Busy time
        self.is_broken = False
        self.available_time = 0.0
        self.schedule_history = [] 

    def update_busy_time(self, duration):
        self.T_k += duration

    def set_broken(self, repair_duration, breakdown_start):
        self.is_broken = True
        self.rho_k += 1
        self.available_time = max(self.available_time, breakdown_start + repair_duration)

    def repair_completed(self):
        self.is_broken = False

# --- 3. FACTORY CLASS ---
# (Giữ nguyên logic Breakdown bạn đã viết, chỉ update tên biến cho khớp)
class Factory:
    def __init__(self, parameters: Parameter, machines: list, jobs: list):
        self.params = parameters
        self.machines = machines
        self.jobs = jobs
        
    @property
    def total_busy_time_R(self):
        return sum(m.T_k for m in self.machines)

    @property
    def total_repairs_rho(self):
        return sum(m.rho_k for m in self.machines)

    def update_machine_states(self, current_makespan):
        """Tính toán hỏng hóc theo Eq. 22, 23, 24"""
        R = self.total_busy_time_R if self.total_busy_time_R > 0 else 1.0
        rho = self.total_repairs_rho if self.total_repairs_rho > 0 else 1.0

        for m in self.machines:
            if m.is_broken: continue
            
            # Eq. 22: Breakdown Probability
            term1 = self.params.lambda_1 * (m.T_k / R)
            term2 = self.params.lambda_2 * m.v
            term3 = self.params.lambda_3 * (m.rho_k / rho)
            Pk = self.params.lambda_0 * (1 + term1 + term2 + term3)
            
            if random.random() < Pk:
                # Eq. 23: Start Time
                omega = random.random()
                factor = (self.params.alpha_1 + 
                          self.params.alpha_2 * (m.rho_k / rho) +
                          self.params.alpha_3 * (m.T_k / R) +
                          self.params.alpha_4 * omega)
                breakdown_start = current_makespan * factor
                
                # Eq. 24: Repair Time
                epsilon = random.uniform(-self.params.gamma, self.params.gamma)
                base_repair = (self.params.beta_0 + 
                               self.params.beta_1 * m.v + 
                               self.params.beta_2 * (m.rho_k / rho))
                repair_time = base_repair * (1 + epsilon)
                
                m.set_broken(repair_time, breakdown_start)
                print(f"Machine {m.machine_id} Broken! Start: {breakdown_start:.2f}, Duration: {repair_time:.2f}")

# --- 4. OPERATION CLASS (SỬA LẠI QUAN TRỌNG) ---
class Operation:
    def __init__(self, job_id, op_id):
        self.job_id = job_id
        self.op_id = op_id
        
        # Dict lưu thông tin máy khả dụng. 
        # Key: machine_id
        # Value: Dictionary {PT, AP, ST, AS}
        self.compatible_machines = {} 
        
        # Để hỗ trợ thuật toán di truyền, ta cần list ID máy đã sort
        self.sorted_machine_ids = []

    def add_machine_info(self, machine_id, PT, AP, ST, AS):
        """
        Thêm thông tin máy làm được operation này.
        PT: Processing Time (PT_ijk)
        AP: Processing Energy (AP_ijk)
        ST: Setup Time (ST_ijk)
        AS: Setup Energy (AS_ijk)
        """
        self.compatible_machines[machine_id] = {
            'PT': PT,
            'AP': AP,
            'ST': ST,
            'AS': AS
        }
        self.sorted_machine_ids.append(machine_id)
        self.sorted_machine_ids.sort()

# --- 5. JOB CLASS ---
class Job:
    def __init__(self, job_id):
        self.job_id = job_id
        self.operations = []