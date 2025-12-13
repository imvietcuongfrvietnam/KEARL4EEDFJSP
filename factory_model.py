import random
import numpy as np

# ==========================================
# 1. PARAMETER CLASS
# ==========================================
class Parameter:
    """
    Lưu trữ toàn bộ tham số cấu hình của bài toán và môi trường.
    """
    def __init__(self):
        self.n = 0  # Số lượng Job
        self.m = 0  # Số lượng Máy
        
        # --- Energy Coefficients ---
        self.AC = 10.0      # Common Energy consumption per unit time (Eq. 8)
        self.UT_k = 2.0     # Transport Energy consumption per unit time
        self.LI = 1e9       # Số lớn vô cùng (Big M)
        
        # --- Breakdown Parameters (Dựa trên Eq. 22-24) ---
        # Xác suất hỏng (Eq. 22)
        self.lambda_0 = 0.01
        self.lambda_1 = 0.5
        self.lambda_2 = 0.3
        self.lambda_3 = 0.2
        
        # Thời điểm bắt đầu hỏng (Eq. 23)
        self.alpha_1 = 0.2
        self.alpha_2 = 0.3
        self.alpha_3 = 0.3
        self.alpha_4 = 0.2
        
        # Thời gian sửa chữa (Eq. 24)
        self.beta_0 = 5.0
        self.beta_1 = 0.5
        self.beta_2 = 0.5
        self.gamma = 0.1
        
        # --- Environment Data ---
        # Ma trận thời gian vận chuyển [m][m]. Quan trọng cho Eq. 6 (REC)
        self.TT_matrix = [] 

# ==========================================
# 2. MACHINE CLASS
# ==========================================
class Machine:
    def __init__(self, machine_id, age=0, energy_idle_unit=1.0):
        self.machine_id = machine_id
        
        # --- Static Parameters ---
        self.AI = energy_idle_unit # AI_k: Idle Power (Eq. 7)
        
        # --- Dynamic State (Breakdown logic) ---
        self.v = age            # Tuổi của máy (Age)
        self.rho_k = 0          # Số lần đã sửa chữa (Repair count)
        self.T_k = 0.0          # Tổng thời gian bận (Busy time)
        self.is_broken = False  # Trạng thái hiện tại
        self.available_time = 0.0 # Thời điểm máy rảnh tiếp theo
        
        # [QUAN TRỌNG] Lưu lịch sử hỏng để Individual.decode() đọc được
        # List các dict: [{'start': 10, 'end': 15}, ...]
        self.breakdown_history = [] 

    def update_busy_time(self, duration):
        """Cộng dồn thời gian máy chạy để tính xác suất hỏng."""
        self.T_k += duration

    def set_broken(self, repair_duration, breakdown_start):
        """
        Kích hoạt trạng thái hỏng máy.
        Lưu vết vào breakdown_history để thuật toán lập lịch né ra.
        """
        self.is_broken = True
        self.rho_k += 1 # Tăng số lần sửa
        
        end_time = breakdown_start + repair_duration
        
        # Cập nhật thời gian rảnh của máy (nếu đang rảnh thì bị đẩy lùi lại)
        self.available_time = max(self.available_time, end_time)
        
        # Ghi vào lịch sử (Individual sẽ dùng cái này để chèn 'Task hỏng' vào timeline)
        self.breakdown_history.append({
            'start': breakdown_start,
            'end': end_time
        })

    def repair_completed(self):
        """Reset trạng thái sau khi sửa xong (dùng cho simulation)"""
        self.is_broken = False

# ==========================================
# 3. OPERATION CLASS
# ==========================================
class Operation:
    def __init__(self, job_id, op_id):
        self.job_id = job_id
        self.op_id = op_id
        
        # Dict lưu thông tin máy khả dụng. 
        # Key: machine_id, Value: {'PT':..., 'AP':..., 'ST':..., 'AS':...}
        self.compatible_machines = {} 
        
        # List ID máy đã sort để Gene trong NST có thể tham chiếu bằng index (0, 1, 2...)
        self.sorted_machine_ids = []

    def add_machine_info(self, machine_id, PT, AP, ST, AS):
        """
        Thêm thông tin máy làm được operation này.
        - PT: Processing Time
        - AP: Processing Power (Processing Energy Coefficient)
        - ST: Setup Time
        - AS: Setup Power (Setup Energy Coefficient)
        """
        self.compatible_machines[machine_id] = {
            'PT': PT,
            'AP': AP,
            'ST': ST,
            'AS': AS
        }
        # Cập nhật danh sách ID và sort lại ngay để đảm bảo nhất quán
        if machine_id not in self.sorted_machine_ids:
            self.sorted_machine_ids.append(machine_id)
            self.sorted_machine_ids.sort()

# ==========================================
# 4. JOB CLASS
# ==========================================
class Job:
    def __init__(self, job_id):
        self.job_id = job_id
        self.operations = [] # Chứa các đối tượng Operation theo thứ tự

# ==========================================
# 5. FACTORY CLASS
# ==========================================
class Factory:
    """
    Lớp quản lý tổng thể: Chứa Machines, Jobs và Parameters.
    Chịu trách nhiệm kích hoạt sự kiện Dynamic Breakdown.
    """
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
        """
        Kiểm tra và kích hoạt sự cố máy hỏng dựa trên công thức xác suất (Eq. 22).
        Được gọi ở đầu mỗi thế hệ (Generation).
        """
        # Tránh chia cho 0
        R = self.total_busy_time_R if self.total_busy_time_R > 0 else 1.0
        rho = self.total_repairs_rho if self.total_repairs_rho > 0 else 1.0

        for m in self.machines:
            # Nếu máy đang hỏng rồi thì thôi
            if m.is_broken and current_makespan < m.available_time: 
                continue
            
            # Reset cờ hỏng nếu đã qua thời gian sửa
            if m.is_broken and current_makespan >= m.available_time:
                m.repair_completed()

            # --- Tính xác suất hỏng (Eq. 22) ---
            term1 = self.params.lambda_1 * (m.T_k / R)
            term2 = self.params.lambda_2 * m.v
            term3 = self.params.lambda_3 * (m.rho_k / rho)
            Pk = self.params.lambda_0 * (1 + term1 + term2 + term3)
            
            # Random xem có hỏng không
            if random.random() < Pk:
                # --- Tính thời điểm bắt đầu hỏng (Eq. 23) ---
                omega = random.random()
                factor = (self.params.alpha_1 + 
                          self.params.alpha_2 * (m.rho_k / rho) +
                          self.params.alpha_3 * (m.T_k / R) +
                          self.params.alpha_4 * omega)
                
                # Thời điểm hỏng phải nằm trong khoảng Makespan hiện tại
                breakdown_start = current_makespan * factor
                
                # --- Tính thời gian sửa chữa (Eq. 24) ---
                epsilon = random.uniform(-self.params.gamma, self.params.gamma)
                base_repair = (self.params.beta_0 + 
                               self.params.beta_1 * m.v + 
                               self.params.beta_2 * (m.rho_k / rho))
                repair_time = base_repair * (1 + epsilon)
                
                # Cập nhật trạng thái máy
                m.set_broken(repair_time, breakdown_start)
                print(f"[BREAKDOWN] Machine {m.machine_id} hỏng tại {breakdown_start:.1f}, sửa mất {repair_time:.1f}")