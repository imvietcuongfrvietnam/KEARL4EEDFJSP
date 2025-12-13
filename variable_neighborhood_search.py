import random
import copy
import math

class VariableNeighborhoodSearch:
    def __init__(self, factory, tabu_size=10, max_iter=30):
        """
        Knowledge-guided Variable Neighborhood Search (VNS)
        """
        self.factory = factory
        self.tabu_list = [] 
        self.tabu_size = tabu_size
        self.max_iter = max_iter # MNS param (Table 5)

    def run_vns(self, individual):
        """
        Chạy quy trình VNS tuần tự: N1' -> N2' -> N3' -> N4'
        """
        # Đảm bảo có thông tin lịch trình (detailed_schedule)
        if not individual.detailed_schedule or individual.makespan == 0:
            individual.decode()

        best_ind = copy.deepcopy(individual)

        # 1. N1': Critical Path Move (Tabu Search)
        ind_n1 = self.operator_n1_tabu_search(best_ind)
        if ind_n1.makespan < best_ind.makespan:
            best_ind = ind_n1

        # Cập nhật đường găng và block mới sau khi chạy N1
        blocks = self.get_critical_blocks(best_ind)
        if not blocks: return best_ind

        # 2. N2': Move to Block Tail
        ind_n2 = self.operator_n2_block_tail(best_ind, blocks)
        if ind_n2.makespan < best_ind.makespan:
            best_ind = ind_n2
            blocks = self.get_critical_blocks(best_ind) # Update blocks nếu cải thiện

        # 3. N3': Move to Block Head
        ind_n3 = self.operator_n3_block_head(best_ind, blocks)
        if ind_n3.makespan < best_ind.makespan:
            best_ind = ind_n3
            blocks = self.get_critical_blocks(best_ind)

        # 4. N4': Random Swap within Block
        ind_n4 = self.operator_n4_random_swap(best_ind, blocks)
        if ind_n4.makespan < best_ind.makespan:
            best_ind = ind_n4

        return best_ind

    # ========================================================
    #       ALGORITHM 1: FINDING CRITICAL OPERATIONS
    # ========================================================
    def get_critical_path(self, individual):
        """
        Tìm đường găng dựa trên Algorithm 1.
        Sử dụng 'detailed_schedule' từ class Individual.
        """
        schedule = individual.detailed_schedule
        
        # 1. Tìm Operation kết thúc cuối cùng (xác định Makespan)
        critical_op_node = None
        max_end_time = -1.0
        
        for m_id, tasks in schedule.items():
            if not tasks: continue
            last_task = tasks[-1] # Task cuối cùng trên máy
            if last_task['end'] > max_end_time:
                max_end_time = last_task['end']
                critical_op_node = last_task

        if critical_op_node is None: return []

        # 2. Backtrack (Truy vết ngược)
        critical_path = [critical_op_node]
        current_node = critical_op_node
        
        while True:
            # Điều kiện dừng: Start Time ~ 0
            if current_node['start'] <= 1e-4:
                break
                
            op_obj = current_node['op']
            machine_id = current_node['machine']
            
            # --- Tìm các Tiền nhiệm (Predecessors) ---
            
            # A. Machine Predecessor (Op ngay trước trên cùng máy)
            mach_pred_node = None
            tasks_on_machine = schedule[machine_id]
            # (Giả định tasks đã sort theo start time trong decode)
            # Tìm index của op hiện tại
            # Lưu ý: So sánh object identity để an toàn
            for i in range(len(tasks_on_machine) - 1, 0, -1):
                if tasks_on_machine[i]['op'] is op_obj:
                    mach_pred_node = tasks_on_machine[i-1]
                    break
            
            # B. Job Predecessor (Op trước của cùng Job)
            job_pred_node = None
            job_pred_finish_time = 0.0
            
            if op_obj.op_id > 0: # Không phải op đầu tiên của Job
                # Tìm Op trước trong Job (cần tìm nó chạy máy nào)
                pred_op_obj = individual.jobs[op_obj.job_id].operations[op_obj.op_id - 1]
                
                # Quét schedule để tìm node của pred_op_obj
                # (Có thể tối ưu bằng map ngược, nhưng loop này cũng nhanh)
                found = False
                for m_id, m_tasks in schedule.items():
                    for t in m_tasks:
                        if t['op'] is pred_op_obj:
                            job_pred_node = t
                            found = True
                            break
                    if found: break
                
                # Tính thời gian sau khi vận chuyển
                if job_pred_node:
                    transport = 0.0
                    if job_pred_node['machine'] != machine_id:
                        transport = self.factory.params.TT_matrix[job_pred_node['machine']][machine_id]
                    job_pred_finish_time = job_pred_node['end'] + transport

            # --- So sánh để chọn hướng đi tiếp (Algorithm 1) ---
            # StartTime hiện tại bị ràng buộc bởi cái nào kết thúc muộn nhất?
            
            mach_pred_end = mach_pred_node['end'] if mach_pred_node else 0.0
            
            # Ưu tiên Machine Constraint nếu thời gian khít nhau (để tạo Block)
            # Dung sai so sánh số thực
            EPS = 1e-4
            
            is_machine_constrained = abs(current_node['start'] - mach_pred_end) < EPS
            is_job_constrained = abs(current_node['start'] - job_pred_finish_time) < EPS
            
            if is_machine_constrained:
                critical_path.append(mach_pred_node)
                current_node = mach_pred_node
            elif is_job_constrained and job_pred_node:
                critical_path.append(job_pred_node)
                current_node = job_pred_node
            else:
                # Trường hợp đặc biệt (ví dụ Job đến sớm nhưng phải đợi Setup/Idle)
                # Theo heuristic: ưu tiên bám theo máy nếu có thể để hình thành block dài
                if mach_pred_node and current_node['start'] >= mach_pred_end:
                     critical_path.append(mach_pred_node)
                     current_node = mach_pred_node
                elif job_pred_node:
                     critical_path.append(job_pred_node)
                     current_node = job_pred_node
                else:
                    break # Không tìm thấy đường

        return critical_path[::-1] # Đảo ngược lại [Start -> End]

    def get_critical_blocks(self, individual):
        """
        Gom các Critical Op liên tiếp trên cùng 1 máy thành Block.
        """
        path = self.get_critical_path(individual)
        if not path: return []
        
        blocks = []
        current_block = [path[0]]
        
        for i in range(1, len(path)):
            prev = path[i-1]
            curr = path[i]
            
            if prev['machine'] == curr['machine']:
                current_block.append(curr)
            else:
                if len(current_block) > 1: # Block chỉ tính khi có >= 2 ops
                    blocks.append(current_block)
                current_block = [curr]
        
        if len(current_block) > 1:
            blocks.append(current_block)
            
        return blocks

    # ========================================================
    #       NEIGHBORHOOD OPERATORS (N1 - N4)
    # ========================================================

    def operator_n1_tabu_search(self, individual):
        """
        N1': Critical path movement (Tabu Search).
        Thay đổi máy cho 1 operation trên đường găng.
        """
        path = self.get_critical_path(individual)
        # Chỉ xét op có thể chuyển sang máy khác
        candidates = [node for node in path if len(node['op'].sorted_machine_ids) > 1]
        
        if not candidates: return individual
        
        best_global_ind = copy.deepcopy(individual)
        curr_ind = copy.deepcopy(individual)
        
        # Tabu Loop
        for _ in range(self.max_iter):
            # Chọn ngẫu nhiên 1 op để di chuyển
            target_node = random.choice(candidates)
            op_obj = target_node['op']
            
            # Lấy index gene trong MS (O(1) nhờ map)
            gene_idx = curr_ind.op_to_index_map[(op_obj.job_id, op_obj.op_id)]
            current_ms_val = curr_ind.ms[gene_idx]
            
            num_machines = len(op_obj.sorted_machine_ids)
            possible_moves = range(num_machines)
            
            best_local_ind = None
            best_local_move_info = None # (gene_idx, new_val, signature)
            min_local_makespan = float('inf')
            
            # Thử di chuyển sang tất cả các máy khác
            for new_val in possible_moves:
                if new_val == current_ms_val: continue
                
                # Tabu check: (Job, Op, NewMachineIndex)
                move_sig = (op_obj.job_id, op_obj.op_id, new_val)
                
                # Aspiration Criteria: Nếu bị cấm nhưng tốt hơn Global Best thì vẫn lấy
                is_tabu = move_sig in self.tabu_list
                
                # Tạo neighbor
                temp_ind = copy.deepcopy(curr_ind)
                temp_ind.ms[gene_idx] = new_val
                temp_ind.decode() # Tính Makespan
                
                # Logic Aspiration
                if is_tabu and temp_ind.makespan >= best_global_ind.makespan:
                    continue # Skip if tabu and not executing aspiration
                
                if temp_ind.makespan < min_local_makespan:
                    min_local_makespan = temp_ind.makespan
                    best_local_ind = temp_ind
                    best_local_move_info = (move_sig)

            # Thực hiện move tốt nhất tìm được
            if best_local_ind:
                curr_ind = best_local_ind
                move_sig = best_local_move_info
                
                # Update Tabu List
                self.tabu_list.append(move_sig)
                if len(self.tabu_list) > self.tabu_size:
                    self.tabu_list.pop(0)
                
                # Update Global Best
                if curr_ind.makespan < best_global_ind.makespan:
                    best_global_ind = copy.deepcopy(curr_ind)
            else:
                break # Dead end

        return best_global_ind

    def operator_n2_block_tail(self, individual, blocks):
        """
        N2': Move op to Block Tail.
        Swap 1 op bất kỳ trong block (trừ tail) ra vị trí cuối block trong OS.
        """
        if not blocks: return individual
        block = random.choice(blocks)
        
        # Cần block có ít nhất 2 phần tử
        if len(block) < 2: return individual
        
        # Chọn op để di chuyển (không chọn tail)
        # Trong bài báo nói "intermediate", nhưng logic tổng quát là move cái gì đó về đuôi
        target_idx = random.randint(0, len(block) - 2)
        target_op = block[target_idx]['op']
        tail_op = block[-1]['op']
        
        return self._swap_ops_in_os(individual, target_op, tail_op)

    def operator_n3_block_head(self, individual, blocks):
        """
        N3': Move op to Block Head.
        Swap 1 op bất kỳ trong block (trừ head) ra vị trí đầu block trong OS.
        """
        if not blocks: return individual
        block = random.choice(blocks)
        if len(block) < 2: return individual
        
        # Chọn op để di chuyển (từ vị trí 1 trở đi)
        target_idx = random.randint(1, len(block) - 1)
        target_op = block[target_idx]['op']
        head_op = block[0]['op']
        
        return self._swap_ops_in_os(individual, target_op, head_op)

    def operator_n4_random_swap(self, individual, blocks):
        """
        N4': Random Swap within Block.
        Tráo đổi vị trí 2 op bất kỳ trong cùng 1 block.
        """
        if not blocks: return individual
        block = random.choice(blocks)
        if len(block) < 2: return individual
        
        idx1, idx2 = random.sample(range(len(block)), 2)
        op1 = block[idx1]['op']
        op2 = block[idx2]['op']
        
        return self._swap_ops_in_os(individual, op1, op2)

    # ================= HELPER FUNCTIONS =================

    def _swap_ops_in_os(self, individual, op1, op2):
        """
        Tìm và hoán đổi vị trí của op1 và op2 trong vector OS.
        Lưu ý: OS chỉ chứa Job ID, cần đếm lần xuất hiện để tìm đúng index.
        """
        new_ind = copy.deepcopy(individual)
        os_vec = new_ind.os
        
        idx1 = self._find_os_index(os_vec, op1)
        idx2 = self._find_os_index(os_vec, op2)
        
        if idx1 != -1 and idx2 != -1:
            # Swap
            os_vec[idx1], os_vec[idx2] = os_vec[idx2], os_vec[idx1]
            new_ind.decode() # Tính lại fitness
            
            # Acceptance Criterion: Chỉ lấy nếu tốt hơn (Greedy)
            if new_ind.makespan < individual.makespan:
                return new_ind
                
        return individual

    def _find_os_index(self, os_vec, op_obj):
        """
        Tìm index của op_obj trong vector OS.
        Vector OS: [1, 2, 1, 3...]
        Op: Job 1, op_id 1 -> Tìm số 1 xuất hiện lần thứ 2.
        """
        target_job = op_obj.job_id
        target_count = op_obj.op_id # 0-based index
        
        current_count = 0
        for i, job_id in enumerate(os_vec):
            if job_id == target_job:
                if current_count == target_count:
                    return i
                current_count += 1
        return -1