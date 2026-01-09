import os
import numpy as np
from factory_model import Parameter, Machine, Job, Operation, Factory

class DataLoader:
    def __init__(self, data_folder):
        self.data_folder = data_folder

    def load_instance(self, instance_name):
        print(f"--- Đang tải dữ liệu: {instance_name} ---")
        
        # 1. Đường dẫn file (Lưu ý tên file phải khớp chính xác)
        path_main = os.path.join(self.data_folder, f"{instance_name}.fjs")
        path_st   = os.path.join(self.data_folder, f"setup_time_{instance_name}.txt")
        path_pe   = os.path.join(self.data_folder, f"process_energy_{instance_name}.txt")
        path_se   = os.path.join(self.data_folder, f"setup_energy_{instance_name}.txt")
        path_idle = os.path.join(self.data_folder, f"idle_energy_{instance_name}.txt")
        path_tran = os.path.join(self.data_folder, f"transport_{instance_name}.txt")

        # Check file gốc
        if not os.path.exists(path_main):
            raise FileNotFoundError(f"[LỖI] Không tìm thấy file: {path_main}")

        # 2. Đọc nội dung
        with open(path_main, 'r') as f:
            lines_pt = [l.strip() for l in f.readlines() if l.strip()]
        
        lines_st = []
        if os.path.exists(path_st):
            with open(path_st, 'r') as f:
                lines_st = [l.strip() for l in f.readlines() if l.strip()]
        else:
            print(f"[Cảnh báo] Thiếu file setup time: {path_st}. Sẽ dùng mặc định.")

        # Load Matrix & Vector
        matrix_pe = self._load_matrix_data(path_pe)
        matrix_se = self._load_matrix_data(path_se)
        idle_vector = self._load_vector(path_idle)
        tt_matrix = self._load_matrix_data(path_tran)

        # 3. Parse Header
        try:
            tokens = lines_pt[0].strip().split()

            # Chỉ lấy 2 phần tử đầu tiên và ép kiểu int
            num_jobs = int(tokens[0])
            num_machines = int(tokens[1])
        except Exception as e:
            raise ValueError(f"Lỗi đọc dòng đầu file .fjs: {e}")

        if not tt_matrix: 
             tt_matrix = [[1.0] * num_machines for _ in range(num_machines)]

        # --- Init Factory ---
        params = Parameter()
        params.n = num_jobs
        params.m = num_machines
        params.TT_matrix = tt_matrix
        params.UT_k = 2.0 
        
        machines = []
        for k in range(num_machines):
            val = idle_vector[k] if k < len(idle_vector) else 1.0
            machines.append(Machine(machine_id=k, energy_idle_unit=val))

        # --- Parse Jobs ---
        jobs = []
        global_op_counter = 0 
        
        for i in range(num_jobs):
            # File PT có header -> dòng dữ liệu là i+1
            if i + 1 >= len(lines_pt): break
            line_pt_vals = list(map(float, lines_pt[i+1].split()))
            
            line_st_vals = []
            if i < len(lines_st):
                line_st_vals = list(map(float, lines_st[i].split()))

            current_job = Job(job_id=i)
            
            total_ops = int(line_pt_vals[0])
            ptr = 1 # Bỏ qua số lượng Ops ở đầu
            
            ptr_st = 1 
            
            for j in range(total_ops):
                op = Operation(job_id=i, op_id=j)
                
                num_compatible = int(line_pt_vals[ptr]) 
                ptr += 1
                if line_st_vals: ptr_st += 1 # Nhảy qua số lượng máy
                
                for _ in range(num_compatible):
                    # Data from .fjs
                    m_id_raw = int(line_pt_vals[ptr])
                    pt_val = line_pt_vals[ptr+1]
                    m_id = m_id_raw - 1
                    
                    # Data from ST
                    st_val = 0.5
                    if line_st_vals and ptr_st+1 < len(line_st_vals):
                        st_val = line_st_vals[ptr_st+1]
                    
                    # Data from Energy Matrix
                    pe_val = 4.0
                    se_val = 2.0
                    if matrix_pe and global_op_counter < len(matrix_pe):
                        if m_id < len(matrix_pe[global_op_counter]):
                            pe_val = matrix_pe[global_op_counter][m_id]
                    if matrix_se and global_op_counter < len(matrix_se):
                        if m_id < len(matrix_se[global_op_counter]):
                            se_val = matrix_se[global_op_counter][m_id]

                    op.add_machine_info(m_id, PT=pt_val, AP=pe_val, ST=st_val, AS=se_val)
                    
                    ptr += 2
                    if line_st_vals: ptr_st += 2
                
                global_op_counter += 1
                current_job.operations.append(op)
            
            jobs.append(current_job)

        return Factory(params, machines, jobs), jobs

    def _load_matrix_data(self, path):
        if not os.path.exists(path): return []
        with open(path, 'r') as f:
            return [list(map(float, l.split())) for l in f if l.strip()]

    def _load_vector(self, path):
        if not os.path.exists(path): return []
        with open(path, 'r') as f:
            return list(map(float, f.read().split()))