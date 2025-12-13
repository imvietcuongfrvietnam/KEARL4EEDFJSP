import os
from data_loader import DataLoader
from kearl_framework import KEARL_Framework

# --- CẤU HÌNH QUAN TRỌNG ---
# Đảm bảo đường dẫn này đúng với máy bạn
DATA_DIR = "./data"  
INSTANCE = "mk01"    

def main():
    # 1. Kiểm tra folder
    print(f"Kiểm tra thư mục: {os.path.abspath(DATA_DIR)}")
    if not os.path.exists(DATA_DIR):
        print(">>> LỖI: Không tìm thấy thư mục 'data'. Hãy tạo folder và copy file vào.")
        return

    # 2. Load Dữ liệu (Không dùng try-except để xem lỗi thật)
    loader = DataLoader(DATA_DIR)
    factory, jobs = loader.load_instance(INSTANCE)

    print(f"-> LOAD THÀNH CÔNG: {len(jobs)} Jobs.")

    # 3. Khởi tạo KEARL
    algorithm = KEARL_Framework(
        factory=factory,
        jobs=jobs,
        pop_size=50,       
        max_gen=50,        
        vns_enabled=True,
        energy_strategy_enabled=True
    )

# 3. Chạy thuật toán
    print("-> Bắt đầu chạy thuật toán...")
    pareto_front = algorithm.run() # Đổi tên biến để rõ nghĩa hơn

    # --- SỬA ĐOẠN NÀY ---
    # Vì kết quả trả về là một danh sách, ta cần chọn ra cá thể có Makespan nhỏ nhất để in
    if isinstance(pareto_front, list):
        # Sắp xếp theo Makespan tăng dần và lấy phần tử đầu tiên
        best_ind = sorted(pareto_front, key=lambda x: x.makespan)[0]
    else:
        best_ind = pareto_front

    # 4. Xuất kết quả
    print("\n" + "="*40)
    print(" KẾT QUẢ TỐI ƯU (BEST MAKESPAN SOLUTION)")
    print("="*40)
    print(f"1. Makespan (MCT):       {best_ind.makespan:.2f}")
    print(f"2. Total Energy (TEC):   {best_ind.total_energy:.2f}")
    print(f"3. Max Workload (WCM):   {best_ind.wcm:.2f}")
    
    # In lịch trình của Job 1 (để kiểm tra)
    print("\nVí dụ lịch trình Job 1:")
    job1_tasks = []
    # Cần kiểm tra xem detailed_schedule có dữ liệu không
    if hasattr(best_ind, 'detailed_schedule') and best_ind.detailed_schedule:
        for m_id, tasks in best_ind.detailed_schedule.items():
            for t in tasks:
                if t['op'].job_id == 0: # Job 0 (tức Job 1)
                    job1_tasks.append(t)
        
        # Sort theo thời gian bắt đầu
        job1_tasks.sort(key=lambda x: x['start'])
        for t in job1_tasks:
            # Op ID + 1 và Machine ID + 1 để khớp với đề bài (từ 1, ko phải từ 0)
            print(f"  Op {t['op'].op_id + 1}: Machine {t['machine'] + 1} | {t['start']:.1f} -> {t['end']:.1f}")
    else:
        print(" (Chưa có dữ liệu lịch trình chi tiết - cần gọi decode() nếu chưa có)")

if __name__ == "__main__":
    main()