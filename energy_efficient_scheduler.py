import copy

class EnergyEfficientScheduler:
    def __init__(self, factory):
        """
        Energy Efficient Scheduling Strategy (Algorithm 3).
        """
        self.factory = factory

    def apply_energy_strategy(self, pareto_front, zz_rate, xx_rate):
        """
        Áp dụng chiến lược ES1, ES2, ES3 cho tập lời giải Pareto.
        
        Args:
            pareto_front (list): Danh sách các Individual không bị trội.
            zz_rate (float): Tỷ lệ phần trăm cho ES1 (Ví dụ 0.3).
            xx_rate (float): Tỷ lệ phần trăm giới hạn cho ES2 (Ví dụ 0.7).
                             (Phần còn lại > xx sẽ dùng ES3).
        """
        n = len(pareto_front)
        # Chuyển đổi tỷ lệ thành chỉ số index
        zz_idx = int(n * zz_rate)
        xx_idx = int(n * xx_rate)

        updated_front = []

        for i in range(n):
            current_ind = pareto_front[i]
            
            # Đảm bảo đã decode để có thông tin Makespan/Energy
            if current_ind.makespan == 0:
                current_ind.decode()

            # Algorithm 3 Logic
            improved_ind = None

            # --- Partition 1: Min Makespan (ES1) ---
            if i < zz_idx:
                # Perform ES1: Shift to machine with min (Setup + Processing)
                improved_ind = self.perform_es1(current_ind)
                
                # Update if Makespan is reduced
                if improved_ind.makespan < current_ind.makespan:
                    updated_front.append(improved_ind)
                else:
                    updated_front.append(current_ind)

            # --- Partition 2: Min Total Energy (ES2) ---
            elif i < xx_idx: # zz <= i < xx
                # Perform ES2: Shift to machine with min Energy (Trans + Setup + Proc)
                improved_ind = self.perform_es2(current_ind)
                
                # Update if Total Energy is reduced
                if improved_ind.total_energy < current_ind.total_energy:
                    updated_front.append(improved_ind)
                else:
                    updated_front.append(current_ind)

            # --- Partition 3: Min Critical Machine Workload (ES3) ---
            else: # i >= xx
                # Perform ES3: Shift to machine with Lowest Workload
                improved_ind = self.perform_es3(current_ind)
                
                # Tính WCM cho cả 2 để so sánh (Eq. 3)
                wcm_current = self._calculate_wcm(current_ind)
                wcm_improved = self._calculate_wcm(improved_ind)

                # Update if Workload is reduced
                if wcm_improved < wcm_current:
                    updated_front.append(improved_ind)
                else:
                    updated_front.append(current_ind)

        return updated_front

    # ========================================================
    #       STRATEGIES (ES1, ES2, ES3) implementation
    #       Target: "The last operation"
    # ========================================================

    def _find_last_operation(self, individual):
        """Helper: Tìm operation kết thúc cuối cùng (quyết định Makespan)."""
        # Nếu chưa có schedule thì decode
        if not individual.detailed_schedule:
            individual.decode()
            
        schedule = individual.detailed_schedule
        last_op_info = None
        max_end_time = -1.0

        for m_id, tasks in schedule.items():
            if not tasks: continue
            last_task = tasks[-1]
            if last_task['end'] > max_end_time:
                max_end_time = last_task['end']
                last_op_info = last_task # {'start', 'end', 'op', 'machine'}

        return last_op_info

    def perform_es1(self, individual):
        """
        ES1: Shift last op to machine with MINIMUM (Setup Time + Processing Time).
        """
        last_op_node = self._find_last_operation(individual)
        if not last_op_node: return individual

        op_obj = last_op_node['op']
        current_m_id = last_op_node['machine']

        # Tìm máy tốt nhất theo tiêu chí ES1
        best_m_idx = -1
        min_time_sum = float('inf')

        # Duyệt qua các máy khả dụng
        for idx, m_id in enumerate(op_obj.sorted_machine_ids):
            # Bỏ qua máy hiện tại? Bài báo không nói rõ, nhưng nên check cả máy khác
            # Lấy thông tin PT, ST
            info = op_obj.compatible_machines[m_id]
            time_sum = info['PT'] + info['ST']

            if time_sum < min_time_sum:
                min_time_sum = time_sum
                best_m_idx = idx

        if best_m_idx != -1:
            new_ind = copy.deepcopy(individual)
            # Cập nhật Gen MS
            gene_idx = new_ind.op_to_index_map[(op_obj.job_id, op_obj.op_id)]
            new_ind.ms[gene_idx] = best_m_idx
            new_ind.decode() 
            return new_ind
        
        return individual

    def perform_es2(self, individual):
        """
        ES2: Transfer last op to machine with SMALLEST Energy Consumption.
        Energy = Transport + Setup + Processing.
        """
        last_op_node = self._find_last_operation(individual)
        if not last_op_node: return individual

        op_obj = last_op_node['op']
        
        prev_m_id = None
        if op_obj.op_id > 0:
            # Tìm op trước
            pred_op = individual.jobs[op_obj.job_id].operations[op_obj.op_id - 1]
            # Tìm trong schedule xem pred_op nằm máy nào
            for m_id, tasks in individual.detailed_schedule.items():
                for t in tasks:
                    if t['op'] is pred_op:
                        prev_m_id = m_id
                        break
                if prev_m_id: break

        best_m_idx = -1
        min_energy = float('inf')

        for idx, m_id in enumerate(op_obj.sorted_machine_ids):
            info = op_obj.compatible_machines[m_id]
            
            # 1. Setup + Processing Energy
            e_proc = info['PT'] * info['AP'] # Eq. 4
            e_setup = info['ST'] * info['AS'] # Eq. 5
            
            # 2. Transport Energy
            e_trans = 0.0
            if prev_m_id and prev_m_id != m_id:
                dist = self.factory.params.TT_matrix[prev_m_id][m_id]
                e_trans = dist * self.factory.params.UT_k # Eq. 6
            
            total_e = e_proc + e_setup + e_trans
            
            if total_e < min_energy:
                min_energy = total_e
                best_m_idx = idx

        if best_m_idx != -1:
            new_ind = copy.deepcopy(individual)
            gene_idx = new_ind.op_to_index_map[(op_obj.job_id, op_obj.op_id)]
            new_ind.ms[gene_idx] = best_m_idx
            new_ind.decode()
            return new_ind
            
        return individual

    def perform_es3(self, individual):
        """
        ES3: Transfer last op to machine with LOWEST Workload.
        Workload = Tổng thời gian bận rộn hiện tại của máy.
        """
        last_op_node = self._find_last_operation(individual)
        if not last_op_node: return individual

        op_obj = last_op_node['op']
        
        # Tính workload hiện tại của các máy
        # (Lưu ý: Workload này tính TRƯỚC khi gán task này hay SAU? 
        # Logic hợp lý: Chọn máy đang rảnh nhất để gán task vào)
        machine_workloads = {}
        for m_id, tasks in individual.detailed_schedule.items():
            # Workload = Tổng duration các task
            load = sum(t['end'] - t['start'] for t in tasks)
            # Trừ đi chính operation này (vì ta đang định di chuyển nó)
            # Nếu op này đang nằm trên máy m_id, ta trừ nó ra để so sánh công bằng
            for t in tasks:
                if t['op'] is op_obj: # So sánh object
                    load -= (t['end'] - t['start'])
            machine_workloads[m_id] = load

        best_m_idx = -1
        min_workload = float('inf')

        for idx, m_id in enumerate(op_obj.sorted_machine_ids):
            # Workload hiện tại của máy
            curr_load = machine_workloads.get(m_id, 0.0)
            
            # Workload dự kiến nếu gán thêm task này
            # (Task size = PT + ST)
            info = op_obj.compatible_machines[m_id]
            added_load = info['PT'] + info['ST']
            
            total_load = curr_load + added_load
            
            if total_load < min_workload:
                min_workload = total_load
                best_m_idx = idx

        if best_m_idx != -1:
            new_ind = copy.deepcopy(individual)
            gene_idx = new_ind.op_to_index_map[(op_obj.job_id, op_obj.op_id)]
            new_ind.ms[gene_idx] = best_m_idx
            new_ind.decode()
            return new_ind
            
        return individual

    def _calculate_wcm(self, individual):
        """
        Helper: Tính Workload of Critical Machine (Eq. 3).
        WCM = Max (Sum PT + Sum ST) over all machines.
        """
        if not individual.detailed_schedule:
            return float('inf')
            
        max_workload = 0.0
        for m_id, tasks in individual.detailed_schedule.items():
            load = sum(t['end'] - t['start'] for t in tasks)
            if load > max_workload:
                max_workload = load
        return max_workload