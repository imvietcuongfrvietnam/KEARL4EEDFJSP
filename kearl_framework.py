import copy
import numpy as np

# Import các module cần thiết
# (Giả sử các file này nằm cùng thư mục)
from initialization import Initialization
from variable_neighborhood_search import VariableNeighborhoodSearch
from energy_efficient_scheduler import EnergyEfficientScheduler
from rl_agent import RLAgent
from nsga2_utils import NSGAII_Utils, nextPopulation

class KEARL_Framework:
    def __init__(self, factory, jobs, 
                 pop_size=100, max_gen=200, 
                 vns_enabled=True, energy_strategy_enabled=True):
        self.factory = factory
        self.jobs = jobs
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.vns_enabled = vns_enabled
        self.es_enabled = energy_strategy_enabled
        
        # [NEW] 1. Khởi tạo list lưu lịch sử hội tụ
        self.convergence_history = [] 
        
        # Modules placeholder
        self.rl_agent = None 
        self.vns = None      
        self.es_scheduler = None 
        
    def run(self):
        print("=== START KEARL ALGORITHM ===")
        
        # 1. Init Modules
        init_module = Initialization(self.pop_size, 0.25, 0.25, 0.25, 0.25, self.jobs, self.factory)
        self.vns = VariableNeighborhoodSearch(self.factory)
        self.es_scheduler = EnergyEfficientScheduler(self.factory)
        self.rl_agent = RLAgent(max_generations=self.max_gen)
        
        # 2. Population Initialization
        print(f"Initializing Population (Size: {self.pop_size})...")
        population = init_module.generate_population()
        
        # Decode & Evaluate Gen 0
        for ind in population:
            ind.decode()
            
        # Init RL State
        current_state = self.rl_agent.get_state(population, 1)

        # Khởi tạo biến lưu trữ Global Best (Tốt nhất lịch sử)
        self.global_best_solution = None
        self.global_min_makespan = float('inf')

        # ================= MAIN EVOLUTIONARY LOOP =================
        for gen in range(1, self.max_gen + 1):
            
            # --- 0. DYNAMIC BREAKDOWN SIMULATION ---
            # Lấy Makespan tốt nhất hiện tại làm mốc thời gian
            current_best_ms = min(ind.makespan for ind in population) if population else 0
            
            # Kiểm tra & Cập nhật hỏng hóc
            self.factory.update_machine_states(current_best_ms)

            # Nếu có breakdown mới, decode lại quần thể cũ để tránh vùng hỏng
            for ind in population:
                ind.decode()

            # --- 3. RL Agent Select Action ---
            Pc, Pm = self.rl_agent.select_action(current_state, gen)
            
            # --- 4. Evolution (Crossover & Mutation) ---
            offspring = nextPopulation(population, Pc, Pm, self.factory)
            
            for ind in offspring:
                ind.decode()
            
            # --- 5. RL Learn ---
            if gen < self.max_gen * 0.8:
                update_method = 'q_learning'
            else:
                update_method = 'sarsa'
            
            self.rl_agent.update_policy(offspring, method=update_method)
            next_state = self.rl_agent.get_state(offspring, gen)
            
            # --- 6. Variable Neighborhood Search (VNS) ---
            combined_pop = population + offspring
            
            if self.vns_enabled:
                fronts = NSGAII_Utils.fast_non_dominated_sort(combined_pop)
                top_front = fronts[0]
                
                limit_vns = min(5, len(top_front))
                for i in range(limit_vns):
                    original_ind = top_front[i]
                    ind_clone = copy.deepcopy(original_ind)
                    
                    improved_ind = self.vns.run_vns(ind_clone)
                    
                    if improved_ind.makespan < original_ind.makespan:
                        if improved_ind.wcm == 0: improved_ind.decode()
                        combined_pop.append(improved_ind)

            # --- 7. Energy Efficient Strategy (ES) ---
            if self.es_enabled:
                fronts = NSGAII_Utils.fast_non_dominated_sort(combined_pop)
                pareto_for_es = fronts[0]
                
                improved_es_list = self.es_scheduler.apply_energy_strategy(
                    pareto_for_es, zz_rate=0.3, xx_rate=0.7
                )
                
                for ind in improved_es_list:
                    if ind.wcm == 0: ind.decode()
                
                combined_pop.extend(improved_es_list)

            # --- 8. Selection (NSGA-II) ---
            population = NSGAII_Utils.select_survivors(combined_pop, self.pop_size)
            
            # --- [NEW] CẬP NHẬT BEST VÀ LỊCH SỬ HỘI TỤ ---
            # Tìm cá thể tốt nhất trong thế hệ hiện tại (theo Makespan)
            current_gen_best = min(population, key=lambda x: x.makespan)
            
            # 1. Lưu vào lịch sử để vẽ biểu đồ
            self.convergence_history.append(current_gen_best.makespan)
            
            # 2. Cập nhật Global Best (Best ever)
            if current_gen_best.makespan < self.global_min_makespan:
                self.global_min_makespan = current_gen_best.makespan
                # Dùng deepcopy để lưu bản cứng, tránh bị biến đổi ở gen sau
                self.global_best_solution = copy.deepcopy(current_gen_best)
            
            current_state = next_state
            
            # Log: In ra cả Best hiện tại (Cur) và Best lịch sử (Hist)
            print(f"Gen {gen}/{self.max_gen} | Cur MS: {current_gen_best.makespan:.1f} | Best Hist: {self.global_min_makespan:.1f} | RL: {update_method}")

        # 9. End
        print("=== END ===")
        final_fronts = NSGAII_Utils.fast_non_dominated_sort(population)
        
        # Trả về 2 giá trị: (Pareto Front cuối cùng, Best Lịch sử)
        return final_fronts[0], self.global_best_solution