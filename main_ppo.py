import os
import sys
import time
import torch
import matplotlib.pyplot as plt  
from data_loader import DataLoader
from kearl_framework import KEARL_Framework
from ppo_agent import PPOAgent # <--- Import PPO đã tạo ở trên

# --- CẤU HÌNH ---
BASE_DATA_DIR = "./data"
INSTANCES_TO_RUN = ["mk05"]
CHART_DIR = "./charts"  

if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)

def draw_convergence_chart(instance_name, history_data):
    if not history_data or len(history_data) == 0:
        print(f"Warning: Không có dữ liệu cho {instance_name}")
        return
    plt.figure(figsize=(10, 6))
    plt.plot(history_data, marker='o', markersize=3, linestyle='-', color='b', label='Best Makespan')
    plt.title(f'Convergence Curve (PPO-KEARL) - {instance_name}')
    plt.xlabel('Generation')
    plt.ylabel('Makespan (Time)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    file_path = os.path.join(CHART_DIR, f"{instance_name}_convergence.png")
    plt.savefig(file_path)
    plt.close()
    print(f"-> Đã lưu biểu đồ tại: {file_path}")

def run_single_instance(instance_name):
    print("\n" + "#"*60)
    print(f"### RUNNING: {instance_name} WITH PPO AGENT ###")
    print("#"*60)

    instance_dir = os.path.join(BASE_DATA_DIR, instance_name)
    if not os.path.exists(instance_dir):
        print(f">>> LỖI: Không tìm thấy thư mục '{instance_dir}'.")
        return None

    try:
        loader = DataLoader(instance_dir)
        factory, jobs = loader.load_instance(instance_name)
        print(f"-> LOAD SUCCESS: {len(jobs)} Jobs.")
    except Exception as e:
        print(f">>> LỖI LOAD DATA: {e}")
        return None

    # --- KHỞI TẠO PPO AGENT ---
    ppo_agent = PPOAgent(max_generations=100)

    # Khởi tạo KEARL và truyền PPO Agent vào
    algorithm = KEARL_Framework(
        factory=factory,
        jobs=jobs,
        pop_size=100,      
        max_gen=100,        
        vns_enabled=True,
        energy_strategy_enabled=True,
        rl_agent=ppo_agent # <--- Đảm bảo KEARL_Framework nhận tham số này
    )

    print(f"-> Bắt đầu chạy KEARL + PPO...")
    start_time = time.time()
    
    final_pareto_front, historical_best = algorithm.run() 
    
    end_time = time.time()
    duration = end_time - start_time

    if hasattr(algorithm, 'convergence_history'):
        draw_convergence_chart(instance_name, algorithm.convergence_history)

    best_ind = historical_best if historical_best is not None else \
               (sorted(final_pareto_front, key=lambda x: x.makespan)[0] if final_pareto_front else None)

    if best_ind:
        print("-" * 40)
        print(f" KẾT QUẢ: {instance_name} - Time: {duration:.2f}s")
        print(f"1. Makespan: {best_ind.makespan:.2f} | 2. Energy: {best_ind.total_energy:.2f}")
        return {"instance": instance_name, "makespan": best_ind.makespan, 
                "energy": best_ind.total_energy, "workload": best_ind.wcm, "time": duration}
    return None

def main():
    summary_results = []
    for instance in INSTANCES_TO_RUN:
        result = run_single_instance(instance)
        if result: summary_results.append(result)

    print("\n\n" + "="*65)
    print(f"{'TỔNG KẾT TOÀN BỘ (PPO-KEARL)':^65}")
    print("="*65)
    print(f"{'Instance':<10} | {'Makespan':<12} | {'Energy':<12} | {'Time(s)':<8}")
    print("-" * 65)
    for res in summary_results:
        print(f"{res['instance']:<10} | {res['makespan']:<12.2f} | {res['energy']:<12.2f} | {res['time']:<8.2f}")
    print("="*65)

if __name__ == "__main__":
    main()