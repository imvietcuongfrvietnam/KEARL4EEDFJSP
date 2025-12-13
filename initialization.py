import random
import math
from individual import Individual
class Initialization:
    def __init__(self, N, random_rate, minimum_rate, maximum_remain_rate, mini_workload_rate, list_job, factory):
        """
        Khởi tạo quần thể ban đầu theo 4 chiến lược.
        
        Args:
            N (int): Kích thước quần thể (Population Size).
            random_rate (float): Tỷ lệ khởi tạo ngẫu nhiên (Strategy 1).
            minimum_rate (float): Tỷ lệ khởi tạo theo thời gian nhỏ nhất (Strategy 2).
            maximum_remain_rate (float): Tỷ lệ khởi tạo theo thời gian còn lại (Strategy 3).
            mini_workload_rate (float): Tỷ lệ khởi tạo theo tải trọng máy nhỏ nhất (Strategy 4).
            list_job (list): Danh sách các đối tượng Job.
            factory (object): Đối tượng Factory chứa thông tin máy.
        """
        self.N = N
        self.list_job = list_job
        self.factory = factory
        
        # Chuẩn hóa tỷ lệ (nếu người dùng nhập không đủ 1.0)
        total_rate = random_rate + minimum_rate + maximum_remain_rate + mini_workload_rate
        self.rates = {
            'random': random_rate / total_rate,
            'min_time': minimum_rate / total_rate,
            'max_remain': maximum_remain_rate / total_rate,
            'min_workload': mini_workload_rate / total_rate
        }

    def generate_population(self):
        """
        Hàm chính để tạo quần thể.
        Returns: List[Individual]
        """
        population = []
        
        # Tính số lượng cá thể cho mỗi chiến lược
        count_random = int(self.N * self.rates['random'])
        count_min_time = int(self.N * self.rates['min_time'])
        count_max_remain = int(self.N * self.rates['max_remain'])
        # Số còn lại cho chiến lược cuối để đảm bảo tổng = N
        count_min_workload = self.N - (count_random + count_min_time + count_max_remain)

        print(f"Initializing Population: Random={count_random}, MinTime={count_min_time}, "
              f"MaxRemain={count_max_remain}, MinWorkload={count_min_workload}")

        # --- 1. Random Strategy ---
        for _ in range(count_random):
            ind = Individual(self.list_job, self.factory, init_strategy=self._strategy_random)
            population.append(ind)

        # --- 2. Minimum Time Strategy ---
        for _ in range(count_min_time):
            ind = Individual(self.list_job, self.factory, init_strategy=self._strategy_min_time)
            population.append(ind)

        # --- 3. Maximum Remaining Time Strategy ---
        for _ in range(count_max_remain):
            ind = Individual(self.list_job, self.factory, init_strategy=self._strategy_max_remain)
            population.append(ind)

        # --- 4. Minimum Workload Strategy ---
        for _ in range(count_min_workload):
            ind = Individual(self.list_job, self.factory, init_strategy=self._strategy_min_workload)
            population.append(ind)

        return population

    # ========================================================
    #               CÁC CHIẾN LƯỢC CỤ THỂ
    # ========================================================

    def _create_random_os(self, individual):
        """Helper: Tạo OS ngẫu nhiên cơ bản"""
        os = []
        for job in self.list_job:
            os.extend([job.job_id] * len(job.operations))
        random.shuffle(os)
        individual.os = os

    def _strategy_random(self, individual, jobs, factory):
        """(1) Generating the initial solutions completely randomly"""
        # Random OS
        self._create_random_os(individual)
        
        # Random MS
        for i, op in enumerate(individual.all_operations):
            num_machines = len(op.compatible_machines)
            individual.ms[i] = random.randint(0, num_machines - 1)

    def _strategy_min_time(self, individual, jobs, factory):
        """(2) Generating according to the minimum time"""
        # Random OS (để giữ sự đa dạng, chỉ tối ưu MS)
        self._create_random_os(individual)

        # Optimize MS: Chọn máy có Processing Time nhỏ nhất
        for i, op in enumerate(individual.all_operations):
            best_idx = -1
            min_pt = float('inf')
            
            # Duyệt qua các máy khả dụng của Operation này
            machine_keys = list(op.compatible_machines.keys())
            for idx, m_id in enumerate(machine_keys):
                pt = op.compatible_machines[m_id]['PT']
                if pt < min_pt:
                    min_pt = pt
                    best_idx = idx
            
            individual.ms[i] = best_idx

    def _strategy_max_remain(self, individual, jobs, factory):
        """(3) Generating according to the maximum remaining time of processing first"""
        # Random MS
        for i, op in enumerate(individual.all_operations):
            individual.ms[i] = random.randint(0, len(op.compatible_machines) - 1)

        # Optimize OS: Sắp xếp Job theo tổng thời gian gia công giảm dần
        # Tính tổng PT trung bình của mỗi Job (vì máy chưa chọn cố định, ta lấy trung bình)
        job_weights = []
        for job in jobs:
            avg_job_pt = 0
            for op in job.operations:
                # Lấy trung bình PT trên các máy khả dụng
                pts = [m['PT'] for m in op.compatible_machines.values()]
                avg_job_pt += sum(pts) / len(pts)
            job_weights.append((job.job_id, avg_job_pt))
        
        # Sắp xếp Job ID theo thời gian giảm dần (Lớn trước)
        job_weights.sort(key=lambda x: x[1], reverse=True)
        sorted_job_ids = [item[0] for item in job_weights]

        # Tạo OS dựa trên thứ tự này
        os = []
        for job_id in sorted_job_ids:
            # Tìm job object để biết số lượng ops
            job_obj = next(j for j in jobs if j.job_id == job_id)
            os.extend([job_id] * len(job_obj.operations))
        
        # Lưu ý: Chiến lược này thường không shuffle lại để giữ ưu tiên "Maximum Remain First"
        individual.os = os

    def _strategy_min_workload(self, individual, jobs, factory):
        """(4) Allocating machine with smallest workload"""
        # Random OS
        self._create_random_os(individual)

        # Optimize MS: Cân bằng tải (Global Load Selection)
        # Tạo bảng theo dõi tải tạm thời của các máy
        machine_loads = {m.machine_id: 0.0 for m in factory.machines}
        
        # Duyệt qua từng Operation (theo thứ tự ngẫu nhiên hoặc tuần tự trong list all_ops)
        # Để công bằng, ta duyệt tuần tự theo index i của vector MS
        for i, op in enumerate(individual.all_operations):
            best_idx = -1
            min_load = float('inf')
            
            machine_keys = list(op.compatible_machines.keys())
            
            # Thử gán vào từng máy khả dụng
            for idx, m_id in enumerate(machine_keys):
                pt = op.compatible_machines[m_id]['PT']
                current_load = machine_loads[m_id]
                
                # Dự kiến tải sau khi gán
                potential_load = current_load + pt
                
                if potential_load < min_load:
                    min_load = potential_load
                    best_idx = idx
            
            # Chốt phương án tốt nhất
            individual.ms[i] = best_idx
            
            # Cập nhật tải máy để dùng cho vòng lặp sau
            selected_m_id = machine_keys[best_idx]
            selected_pt = op.compatible_machines[selected_m_id]['PT']
            machine_loads[selected_m_id] += selected_pt