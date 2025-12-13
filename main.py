import os
import sys
from data_loader import DataLoader
from kearl_framework import KEARL_Framework

# --- CẤU HÌNH QUAN TRỌNG ---
DATA_DIR = "./data"  
INSTANCE = "mk01"    

def main():
    # 1. Kiểm tra folder
    print(f"Kiểm tra thư mục: {os.path.abspath(DATA_DIR)}")
    if not os.path.exists(DATA_DIR):
        print(">>> LỖI: Không tìm thấy thư mục 'data'. Hãy tạo folder và copy file vào.")
        return

    # 2. Load Dữ liệu
    try:
        loader = DataLoader(DATA_DIR)
        factory, jobs = loader.load_instance(INSTANCE)
        print(f"-> LOAD THÀNH CÔNG: {len(jobs)} Jobs.")
    except Exception as e:
        print(f"Lỗi load data: {e}")
        return

    # 3. Khởi tạo KEARL
    algorithm = KEARL_Framework(
        factory=factory,
        jobs=jobs,
        pop_size=100,       
        max_gen=100,        
        vns_enabled=True,
        energy_strategy_enabled=True
    )

    # 4. Chạy thuật toán
    print("-> Bắt đầu chạy thuật toán...")
    
    final_pareto_front, historical_best = algorithm.run() 

    # Ưu tiên lấy historical_best (kết quả tốt nhất từng tìm thấy)
    if historical_best is not None:
        best_ind = historical_best
        source = "Historical Best (Tìm thấy trong quá khứ)"
    else:
        # Dự phòng: Lấy tốt nhất trong quần thể cuối cùng
        if isinstance(final_pareto_front, list) and len(final_pareto_front) > 0:
            best_ind = sorted(final_pareto_front, key=lambda x: x.makespan)[0]
        else:
            best_ind = final_pareto_front
        source = "Final Generation Best (Tốt nhất thế hệ cuối)"

    # 5. Xuất kết quả
    print("\n" + "="*50)
    print(f" KẾT QUẢ TỐI ƯU ({source})")
    print("="*50)
    
    # Kiểm tra an toàn trước khi in
    if best_ind:
        print(f"1. Makespan (MCT):       {best_ind.makespan:.2f}")
        print(f"2. Total Energy (TEC):   {best_ind.total_energy:.2f}")
        print(f"3. Max Workload (WCM):   {best_ind.wcm:.2f}")
        
        # In lịch trình của Job 1 (để kiểm tra)
        print("\nVí dụ lịch trình Job 1:")
        job1_tasks = []
        
        if hasattr(best_ind, 'detailed_schedule') and best_ind.detailed_schedule:
            for m_id, tasks in best_ind.detailed_schedule.items():
                for t in tasks:
                    # Kiểm tra task loại 'operation' và thuộc Job 0
                    if t.get('type') == 'operation' and t['op'].job_id == 0:
                        job1_tasks.append(t)
            
            # Sort theo thời gian bắt đầu
            job1_tasks.sort(key=lambda x: x['start'])
            
            if not job1_tasks:
                print("  (Không tìm thấy task nào của Job 1)")
            
            for t in job1_tasks:
                # Op ID + 1 và Machine ID + 1 để khớp với đề bài (từ 1, ko phải từ 0)
                op_id = t['op'].op_id + 1
                m_id = t['machine'] + 1
                start = t['start']
                end = t['end']
                print(f"  Op {op_id}: Machine {m_id} | {start:6.1f} -> {end:6.1f}")
        else:
            print(" (Chưa có dữ liệu lịch trình chi tiết - cần gọi decode() nếu chưa có)")
    else:
        print("Không tìm thấy lời giải nào.")

if __name__ == "__main__":
    main()