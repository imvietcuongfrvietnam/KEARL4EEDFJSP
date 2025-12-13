import random
import copy
import numpy as np

class Individual:
    def __init__(self, jobs, factory, init_strategy=None):
        """
        Khởi tạo cá thể cho bài toán EEDFJSP (Hỗ trợ Machine Breakdown).
        """
        self.jobs = jobs
        self.factory = factory
        
        # 1. FLATTEN OPERATIONS & TẠO MAP
        self.op_to_index_map = {} 
        self.all_operations = [] 
        
        idx_counter = 0
        for job in self.jobs:
            for op in job.operations:
                self.all_operations.append(op)
                self.op_to_index_map[(job.job_id, op.op_id)] = idx_counter
                
                if not hasattr(op, 'sorted_machine_ids'):
                    op.sorted_machine_ids = sorted(list(op.compatible_machines.keys()))
                
                idx_counter += 1
        
        self.total_ops = len(self.all_operations)
        
        # 2. GENOTYPE
        self.ms = [0] * self.total_ops
        self.os = [0] * self.total_ops

        # 3. KHỞI TẠO
        if init_strategy:
            init_strategy(self, self.jobs, self.factory)
        
        # 4. PHENOTYPE
        self.makespan = 0.0      
        self.total_energy = 0.0  
        self.wcm = 0.0           
        self.fitness = 0.0       
        self.detailed_schedule = {} 

    def decode(self):
        """
        Insertion-based Decoding (Cập nhật xử lý Breakdown).
        """
        # --- A. Reset trạng thái ---
        machine_timelines = {m.machine_id: [] for m in self.factory.machines} 
        machine_end_times = {m.machine_id: 0.0 for m in self.factory.machines}
        
        # [NEW] --- XỬ LÝ BREAKDOWN: CHÈN CÁC KHOẢNG HỎNG VÀO TIMELINE TRƯỚC ---
        # Coi breakdown như một task cố định để thuật toán insertion tự né
        for m in self.factory.machines:
            # Kiểm tra xem máy có lịch sử hỏng hóc không (được cập nhật từ Factory.update_machine_states)
            if hasattr(m, 'breakdown_history') and m.breakdown_history:
                for bd in m.breakdown_history:
                    # bd là dict {'start': float, 'end': float}
                    machine_timelines[m.machine_id].append({
                        'start': bd['start'],
                        'end': bd['end'],
                        'op': None,       # Không thuộc Job nào
                        'type': 'breakdown' # Đánh dấu loại
                    })
                    # Cập nhật thời gian kết thúc của máy nếu breakdown nằm ở cuối
                    if bd['end'] > machine_end_times[m.machine_id]:
                        machine_end_times[m.machine_id] = bd['end']

        # Các biến theo dõi Job
        job_end_times = {j.job_id: 0.0 for j in self.jobs} 
        job_prev_machine = {j.job_id: None for j in self.jobs} 
        job_op_counter = {j.job_id: 0 for j in self.jobs}

        # Các thành phần năng lượng
        E_processing = 0.0 
        E_setup = 0.0      
        E_transport = 0.0  
        E_idle = 0.0       

        # --- B. Vòng lặp giải mã (Duyệt vector OS) ---
        for job_id in self.os:
            op_idx_in_job = job_op_counter[job_id]
            current_job = next(j for j in self.jobs if j.job_id == job_id)
            current_op = current_job.operations[op_idx_in_job]
            job_op_counter[job_id] += 1

            # 1. Xác định Máy
            gene_idx = self.op_to_index_map[(job_id, current_op.op_id)]
            selected_machine_idx = self.ms[gene_idx]
            machine_id = current_op.sorted_machine_ids[selected_machine_idx]
            mach_info = current_op.compatible_machines[machine_id]
            
            PT = mach_info['PT']
            AP = mach_info['AP']
            ST = mach_info['ST']
            AS = mach_info['AS']

            # 2. Tính Arrival Time
            prev_finish = job_end_times[job_id]
            transport_time = 0.0
            prev_m_id = job_prev_machine[job_id]
            
            if prev_m_id is not None and prev_m_id != machine_id:
                transport_time = self.factory.params.TT_matrix[prev_m_id][machine_id]
                E_transport += transport_time * self.factory.params.UT_k 
            
            arrival_time = prev_finish + transport_time

            # 3. Logic Chèn (Insertion Logic) - Tự động né Breakdown
            duration = PT + ST 
            E_processing += AP * PT 
            E_setup += AS * ST      

            # Sắp xếp timeline để tìm khe hở (Bao gồm cả các khoảng Breakdown đã chèn)
            timeline = sorted(machine_timelines[machine_id], key=lambda x: x['start'])
            
            best_start = -1
            found_gap = False
            prev_block_end = 0.0
            
            for block in timeline:
                block_start = block['start']
                gap_size = block_start - prev_block_end
                
                # Khe hở có đủ nhét vừa task không?
                if gap_size >= duration:
                    potential_start = max(prev_block_end, arrival_time)
                    if potential_start + duration <= block_start:
                        best_start = potential_start
                        found_gap = True
                        break 
                prev_block_end = block['end']
            
            if found_gap:
                start_time = best_start
            else:
                # Nếu không có khe, đặt sau task cuối cùng (hoặc sau breakdown cuối cùng)
                start_time = max(machine_end_times[machine_id], arrival_time)

            end_time = start_time + duration
            
            # 4. Cập nhật trạng thái
            task_info = {
                'start': start_time,
                'end': end_time,
                'op': current_op,
                'machine': machine_id,
                'type': 'operation' # Đánh dấu là task thường
            }
            machine_timelines[machine_id].append(task_info)
            machine_end_times[machine_id] = max(machine_end_times[machine_id], end_time)
            job_end_times[job_id] = end_time
            job_prev_machine[job_id] = machine_id

        # --- C. Tính toán Fitness ---
        
        self.makespan = max(machine_end_times.values()) if machine_end_times else 0

        max_machine_workload = 0.0
        
        for m_id, end_time_k in machine_end_times.items():
            timeline = machine_timelines[m_id]
            
            # [SỬA ĐỔI] Tính Workload: Chỉ tính thời gian làm việc thực (Operation), KHÔNG tính Breakdown
            # Workload = Sum(PT + ST)
            busy_duration = sum(
                (t['end'] - t['start']) 
                for t in timeline 
                if t.get('type') == 'operation' # Chỉ cộng task sản xuất
            )
            
            if busy_duration > max_machine_workload:
                max_machine_workload = busy_duration
            
            # Tính Idle Time: Tổng thời gian - Workload - Breakdown_Duration
            # Tuy nhiên công thức đơn giản là: Makespan_Machine - Busy_Duration - Total_Breakdown
            total_breakdown_duration = sum(
                (t['end'] - t['start']) 
                for t in timeline 
                if t.get('type') == 'breakdown'
            )
            
            # Idle time thực tế (máy bật nhưng không chạy và không sửa)
            idle_duration = max(0, end_time_k - busy_duration - total_breakdown_duration)
            
            machine_obj = next(m for m in self.factory.machines if m.machine_id == m_id)
            E_idle += idle_duration * machine_obj.AI

        self.wcm = max_machine_workload
        E_common = self.makespan * self.factory.params.AC
        self.total_energy = E_processing + E_setup + E_transport + E_idle + E_common
        
        self.detailed_schedule = machine_timelines

    # ... (Giữ nguyên các hàm Crossover và Mutation ở dưới) ...
    def crossover_machine_selection(self, partner):
        child1 = Individual(self.jobs, self.factory)
        child2 = Individual(self.jobs, self.factory)
        child1.os = self.os[:]
        child2.os = partner.os[:]
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
        child1 = Individual(self.jobs, self.factory)
        child2 = Individual(self.jobs, self.factory)
        child1.ms = self.ms[:]
        child2.ms = partner.ms[:]
        
        all_job_ids = list(set(self.os))
        subset_size = len(all_job_ids) // 2
        subset_jobs = set(random.sample(all_job_ids, subset_size))

        def apply_jox(parent_main, parent_fill):
            new_os = [-1] * self.total_ops
            for i, job_id in enumerate(parent_main.os):
                if job_id in subset_jobs:
                    new_os[i] = job_id
            fill_idx = 0
            for job_id in parent_fill.os:
                if job_id not in subset_jobs:
                    while new_os[fill_idx] != -1:
                        fill_idx += 1
                    new_os[fill_idx] = job_id
            return new_os

        child1.os = apply_jox(self, partner)
        child2.os = apply_jox(partner, self)
        return child1, child2

    def mutation_machine_selection(self, mutation_rate):
        for i in range(self.total_ops):
            if random.random() < mutation_rate:
                op_obj = self.all_operations[i]
                num_machines = len(op_obj.sorted_machine_ids)
                if num_machines > 1:
                    current_idx = self.ms[i]
                    new_idx = current_idx
                    while new_idx == current_idx:
                        new_idx = random.randint(0, num_machines - 1)
                    self.ms[i] = new_idx

    def mutation_operation_sequence(self, mutation_rate):
        if random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(self.total_ops), 2)
            self.os[idx1], self.os[idx2] = self.os[idx2], self.os[idx1]