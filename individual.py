import random
import copy
import numpy as np

class Individual:
    def __init__(self, jobs, factory, init_strategy=None):
        """
        Khởi tạo cá thể cho bài toán EEDFJSP.
        """
        self.jobs = jobs
        self.factory = factory
        
        # 1. FLATTEN OPERATIONS & TẠO MAP (Tối ưu hóa hiệu năng)
        # Map: (Job_ID, Op_ID) -> Index phẳng trong nhiễm sắc thể
        self.op_to_index_map = {} 
        self.all_operations = [] 
        
        idx_counter = 0
        for job in self.jobs:
            for op in job.operations:
                self.all_operations.append(op)
                # Map để tra cứu ngược nhanh O(1)
                self.op_to_index_map[(job.job_id, op.op_id)] = idx_counter
                
                # Quan trọng: Sort list machine để index trong MS luôn nhất quán
                if not hasattr(op, 'sorted_machine_ids'):
                    op.sorted_machine_ids = sorted(list(op.compatible_machines.keys()))
                
                idx_counter += 1
        
        self.total_ops = len(self.all_operations)
        
        # 2. GENOTYPE (Nhiễm sắc thể)
        # ms: Machine Selection (Lưu index của máy trong sorted_machine_ids)
        # os: Operation Sequence (Lưu Job ID)
        self.ms = [0] * self.total_ops
        self.os = [0] * self.total_ops

        # 3. KHỞI TẠO (Theo chiến lược 4.3 trong bài báo)
        if init_strategy:
            init_strategy(self, self.jobs, self.factory)
        
        # 4. PHENOTYPE (Kết quả sau decode)
        self.makespan = 0.0      # Maximum Completion Time (MCT - Eq. 1)
        self.total_energy = 0.0  # Total Energy Consumption (TEC - Eq. 2)
        self.wcm = 0.0           # Workload of Critical Machine (WCM - Eq. 3) [MỚI]
        self.fitness = 0.0       # Giá trị fitness tổng hợp
        
        # Lưu lịch chi tiết để dùng cho Local Search (VNS) sau này
        self.detailed_schedule = {} 

    def decode(self):
        """
        Insertion-based Decoding.
        Chuyển đổi MS/OS thành lịch sản xuất và tính toán MCT, TEC, WCM.
        """
        # --- A. Reset trạng thái ---
        # machine_timelines: Lưu các khoảng bận {'start', 'end'} trên từng máy
        machine_timelines = {m.machine_id: [] for m in self.factory.machines} 
        machine_end_times = {m.machine_id: 0.0 for m in self.factory.machines}
        
        # job_end_times: Thời điểm hoàn thành op trước đó của Job
        job_end_times = {j.job_id: 0.0 for j in self.jobs} 
        job_prev_machine = {j.job_id: None for j in self.jobs} 
        
        # Biến đếm cục bộ để biết đang xét đến op thứ mấy của job
        job_op_counter = {j.job_id: 0 for j in self.jobs}

        # Các thành phần năng lượng (Eq. 4-8)
        E_processing = 0.0 # PEC
        E_setup = 0.0      # SEC
        E_transport = 0.0  # REC
        E_idle = 0.0       # IEC

        # --- B. Vòng lặp giải mã (Duyệt vector OS) ---
        for job_id in self.os:
            # Lấy thông tin Operation hiện tại
            op_idx_in_job = job_op_counter[job_id]
            current_job = next(j for j in self.jobs if j.job_id == job_id)
            current_op = current_job.operations[op_idx_in_job]
            
            # Tăng biến đếm
            job_op_counter[job_id] += 1

            # 1. Xác định Máy (từ vector MS)
            # Dùng map O(1) thay vì loop O(N)
            gene_idx = self.op_to_index_map[(job_id, current_op.op_id)]
            selected_machine_idx = self.ms[gene_idx]
            
            # Lấy Machine ID thực tế
            machine_id = current_op.sorted_machine_ids[selected_machine_idx]
            mach_info = current_op.compatible_machines[machine_id]
            
            # Lấy thông số PT, Power, Setup
            PT = mach_info['PT']
            AP = mach_info['AP'] # Processing Power
            ST = mach_info['ST'] # Setup Time
            AS = mach_info['AS'] # Setup Power

            # 2. Tính Arrival Time (Constraint 16)
            prev_finish = job_end_times[job_id]
            transport_time = 0.0
            prev_m_id = job_prev_machine[job_id]
            
            if prev_m_id is not None and prev_m_id != machine_id:
                # Lấy thời gian từ ma trận TT
                transport_time = self.factory.params.TT_matrix[prev_m_id][machine_id]
                E_transport += transport_time * self.factory.params.UT_k # REC (Eq. 6)
            
            arrival_time = prev_finish + transport_time

            # 3. Logic Chèn (Insertion Logic)
            duration = PT + ST 
            E_processing += AP * PT # PEC (Eq. 4)
            E_setup += AS * ST      # SEC (Eq. 5)

            # Mặc định: Đặt vào cuối cùng
            start_time = max(machine_end_times[machine_id], arrival_time)
            best_start = -1
            found_gap = False

            # Quét các khe hở (Idle slots) trên máy để chèn
            timeline = sorted(machine_timelines[machine_id], key=lambda x: x['start'])
            prev_block_end = 0.0
            
            for block in timeline:
                block_start = block['start']
                gap_size = block_start - prev_block_end
                
                # Kiểm tra khe có đủ rộng & job đã đến nơi chưa
                if gap_size >= duration:
                    potential_start = max(prev_block_end, arrival_time)
                    if potential_start + duration <= block_start:
                        best_start = potential_start
                        found_gap = True
                        break # First-fit strategy
                prev_block_end = block['end']
            
            if found_gap:
                start_time = best_start
            else:
                # Nếu không có khe nào vừa, đặt sau task cuối cùng
                start_time = max(machine_end_times[machine_id], arrival_time)

            end_time = start_time + duration
            
            # 4. Cập nhật trạng thái
            task_info = {
                'start': start_time,
                'end': end_time,
                'op': current_op,
                'machine': machine_id
            }
            machine_timelines[machine_id].append(task_info)
            machine_end_times[machine_id] = max(machine_end_times[machine_id], end_time)
            job_end_times[job_id] = end_time
            job_prev_machine[job_id] = machine_id

        # --- C. Tính toán Fitness (MCT, WCM, TEC) ---
        
        # 1. Makespan (MCT - Eq. 1)
        self.makespan = max(machine_end_times.values()) if machine_end_times else 0

        # 2. Idle Energy (IEC) & Workload of Critical Machine (WCM)
        max_machine_workload = 0.0
        
        for m_id, end_time_k in machine_end_times.items():
            timeline = machine_timelines[m_id]
            
            # Tổng thời gian máy thực sự chạy (Workload = Sum(PT + ST))
            busy_duration = sum(t['end'] - t['start'] for t in timeline)
            
            # Cập nhật WCM (Eq. 3)
            if busy_duration > max_machine_workload:
                max_machine_workload = busy_duration
            
            # Thời gian nhàn rỗi (IEC - Eq. 7)
            # Idle = Makespan_of_Machine - Busy_Time
            idle_duration = max(0, end_time_k - busy_duration)
            
            # Lấy hệ số AI của máy
            machine_obj = next(m for m in self.factory.machines if m.machine_id == m_id)
            E_idle += idle_duration * machine_obj.AI

        self.wcm = max_machine_workload # Gán giá trị WCM

        # 3. Common Energy (CEC - Eq. 8)
        E_common = self.makespan * self.factory.params.AC

        # 4. Total Energy (TEC - Eq. 2)
        self.total_energy = E_processing + E_setup + E_transport + E_idle + E_common
        
        # Lưu kết quả timeline để vẽ Gantt hoặc chạy VNS
        self.detailed_schedule = machine_timelines

    # =================================================================
    #                           GENETIC OPERATORS
    # =================================================================

    def crossover_machine_selection(self, partner):
        """
        Lai ghép MS: Multi-point Crossover (Fig 3a - Top).
        Dùng mặt nạ ngẫu nhiên để tráo đổi gene máy giữa cha và mẹ.
        """
        child1 = Individual(self.jobs, self.factory)
        child2 = Individual(self.jobs, self.factory)
        
        # OS copy tạm (sẽ lai ghép ở bước sau hoặc giữ nguyên tùy quy trình)
        child1.os = self.os[:]
        child2.os = partner.os[:]
        
        # Tạo mask: 0 lấy từ self, 1 lấy từ partner
        mask = [random.randint(0, 1) for _ in range(self.total_ops)]
        
        for i in range(self.total_ops):
            if mask[i] == 0:
                child1.ms[i] = self.ms[i]
                child2.ms[i] = partner.ms[i]
            else:
                child1.ms[i] = partner.ms[i]
                child2.ms[i] = self.ms[i]
                
        return child1, child2

    def crossover_operation_sequence(self, partner):
        """
        Lai ghép OS: Job-based Crossover (JOX) (Fig 3a - Bottom).
        Giữ cấu trúc Job, không làm vỡ ràng buộc thứ tự precedence.
        """
        child1 = Individual(self.jobs, self.factory)
        child2 = Individual(self.jobs, self.factory)
        
        # MS giữ nguyên (hoặc nhận từ bước crossover MS trước đó)
        child1.ms = self.ms[:]
        child2.ms = partner.ms[:]

        # 1. Chọn tập Job con (Subset Jobs)
        # Lấy danh sách Job ID duy nhất
        all_job_ids = list(set(self.os))
        subset_size = len(all_job_ids) // 2
        # Chọn ngẫu nhiên một nửa số job
        subset_jobs = set(random.sample(all_job_ids, subset_size))

        # Hàm nội bộ thực hiện logic JOX
        def apply_jox(parent_main, parent_fill):
            new_os = [-1] * self.total_ops
            
            # Bước 1: Copy gene thuộc subset_jobs từ Parent Main (giữ nguyên vị trí)
            for i, job_id in enumerate(parent_main.os):
                if job_id in subset_jobs:
                    new_os[i] = job_id
            
            # Bước 2: Điền các gene còn lại từ Parent Fill (giữ nguyên thứ tự xuất hiện)
            fill_idx = 0
            for job_id in parent_fill.os:
                if job_id not in subset_jobs:
                    # Tìm ô trống tiếp theo
                    while new_os[fill_idx] != -1:
                        fill_idx += 1
                    new_os[fill_idx] = job_id
            return new_os

        child1.os = apply_jox(self, partner)
        child2.os = apply_jox(partner, self)
        
        return child1, child2

    def mutation_machine_selection(self, mutation_rate):
        """
        Đột biến MS: Random Machine Selection (Fig 3b - Left).
        Duyệt qua từng gen, nếu dính xác suất thì chọn máy khác ngẫu nhiên.
        """
        for i in range(self.total_ops):
            if random.random() < mutation_rate:
                # Lấy Op object từ index
                # (Vì self.all_operations đã được flatten theo thứ tự index i)
                op_obj = self.all_operations[i]
                num_machines = len(op_obj.sorted_machine_ids)
                
                if num_machines > 1:
                    current_idx = self.ms[i]
                    # Chọn máy mới khác máy cũ
                    new_idx = current_idx
                    while new_idx == current_idx:
                        new_idx = random.randint(0, num_machines - 1)
                    self.ms[i] = new_idx

    def mutation_operation_sequence(self, mutation_rate):
        """
        Đột biến OS: Swap Mutation (Fig 3b - Right).
        Hoán đổi vị trí của 2 operation bất kỳ trong chuỗi OS.
        Lưu ý: Mutation Rate ở đây thường áp dụng cho CẢ CÁ THỂ (Individual).
        """
        if random.random() < mutation_rate:
            # Chọn 2 vị trí ngẫu nhiên bất kỳ
            idx1, idx2 = random.sample(range(self.total_ops), 2)
            
            # Hoán đổi
            self.os[idx1], self.os[idx2] = self.os[idx2], self.os[idx1]