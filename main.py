import os
import sys
import time
import matplotlib.pyplot as plt  
from data_loader import DataLoader
from kearl_framework import KEARL_Framework

# --- CẤU HÌNH ---
BASE_DATA_DIR = "./data"
INSTANCES_TO_RUN = ["mk05"]
CHART_DIR = "./charts"  # <--- [THÊM MỚI] Thư mục lưu ảnh biểu đồ

# Tạo thư mục lưu biểu đồ nếu chưa có
if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)

def draw_convergence_chart(instance_name, history_data):
    """
    [THÊM MỚI] Hàm vẽ biểu đồ hội tụ và lưu thành file ảnh
    Args:
        instance_name: Tên bộ dữ liệu (mk01...)
        history_data: List chứa giá trị Best Makespan qua từng generation
    """
    if not history_data or len(history_data) == 0:
        print(f"Warning: Không có dữ liệu lịch sử để vẽ biểu đồ cho {instance_name}")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(history_data, marker='o', markersize=3, linestyle='-', color='b', label='Best Makespan')
    
    plt.title(f'Convergence Curve - {instance_name}')
    plt.xlabel('Generation')
    plt.ylabel('Makespan (Time)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Lưu file
    file_path = os.path.join(CHART_DIR, f"{instance_name}_convergence.png")
    plt.savefig(file_path)
    plt.close() # Đóng plot để giải phóng bộ nhớ
    print(f"-> Đã lưu biểu đồ tại: {file_path}")

def run_single_instance(instance_name):
    """
    Hàm chạy thử nghiệm cho 1 bộ dữ liệu cụ thể
    """
    print("\n" + "#"*60)
    print(f"### ĐANG XỬ LÝ INSTANCE: {instance_name} ###")
    print("#"*60)

    # 1. Xác định đường dẫn
    instance_dir = os.path.join(BASE_DATA_DIR, instance_name)
    
    if not os.path.exists(instance_dir):
        print(f">>> LỖI: Không tìm thấy thư mục '{instance_dir}'. Bỏ qua.")
        return None

    # 2. Load Dữ liệu
    try:
        loader = DataLoader(instance_dir)
        factory, jobs = loader.load_instance(instance_name)
        print(f"-> LOAD THÀNH CÔNG: {len(jobs)} Jobs từ {instance_name}.")
    except Exception as e:
        print(f">>> LỖI LOAD DATA ({instance_name}): {e}")
        return None

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
    print(f"-> Bắt đầu chạy thuật toán cho {instance_name}...")
    start_time = time.time()
    
    # --- [LƯU Ý] ---
    # Bạn cần đảm bảo algorithm.run() trả về history, hoặc đối tượng algorithm lưu nó.
    # Ở đây tôi giả định algorithm object có thuộc tính 'convergence_history'
    # là một list chứa best makespan của mỗi generation.
    final_pareto_front, historical_best = algorithm.run() 
    
    end_time = time.time()
    duration = end_time - start_time

    # --- [THÊM MỚI] VẼ BIỂU ĐỒ ---
    # Kiểm tra xem thuật toán có thuộc tính lưu lịch sử không
    if hasattr(algorithm, 'convergence_history'):
        draw_convergence_chart(instance_name, algorithm.convergence_history)
    else:
        # Nếu framework của bạn chưa hỗ trợ, hãy thêm một list self.history vào KEARL_Framework
        print("!!! CẢNH BÁO: Không tìm thấy 'convergence_history' trong algorithm object. Không thể vẽ biểu đồ.")

    # 5. Chọn lời giải tốt nhất
    best_ind = None
    source = "N/A"

    if historical_best is not None:
        best_ind = historical_best
        source = "Historical Best"
    else:
        if isinstance(final_pareto_front, list) and len(final_pareto_front) > 0:
            best_ind = sorted(final_pareto_front, key=lambda x: x.makespan)[0]
            source = "Final Generation Best"
        else:
            best_ind = final_pareto_front
            source = "Single Result"

    # 6. In kết quả chi tiết
    if best_ind:
        print("-" * 40)
        print(f" KẾT QUẢ: {instance_name} ({source}) - Chạy trong {duration:.2f}s")
        print("-" * 40)
        print(f"1. Makespan (MCT):       {best_ind.makespan:.2f}")
        print(f"2. Total Energy (TEC):   {best_ind.total_energy:.2f}")
        print(f"3. Max Workload (WCM):   {best_ind.wcm:.2f}")
        
        # print(f"\nVí dụ lịch trình Job 1 ({instance_name}):")
        # print_job1_schedule(best_ind)
        
        return {
            "instance": instance_name,
            "makespan": best_ind.makespan,
            "energy": best_ind.total_energy,
            "workload": best_ind.wcm,
            "time": duration
        }
    else:
        print(">>> KHÔNG TÌM THẤY LỜI GIẢI.")
        return None

def print_job1_schedule(best_ind):
    # (Giữ nguyên code cũ của bạn ở đây)
    pass

def main():
    summary_results = []
    for instance in INSTANCES_TO_RUN:
        result = run_single_instance(instance)
        if result:
            summary_results.append(result)

    # In bảng tổng kết
    print("\n\n")
    print("="*65)
    print(f"{'TỔNG KẾT TOÀN BỘ':^65}")
    print("="*65)
    print(f"{'Instance':<10} | {'Makespan':<12} | {'Energy':<12} | {'Workload':<12} | {'Time(s)':<8}")
    print("-" * 65)
    for res in summary_results:
        print(f"{res['instance']:<10} | {res['makespan']:<12.2f} | {res['energy']:<12.2f} | {res['workload']:<12.2f} | {res['time']:<8.2f}")
    print("="*65)

if __name__ == "__main__":
    main()