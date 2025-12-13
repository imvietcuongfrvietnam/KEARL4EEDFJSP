import copy

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
        
        # Modules
        self.rl_agent = None 
        self.vns = None      
        self.es_scheduler = None 
        
    # Hàm calculate_wcm đã được loại bỏ vì Individual.decode() tự tính toán thuộc tính .wcm

    def run(self):
        print("=== START KEARL ALGORITHM ===")
        
        # 1. Init Modules
        # (Lưu ý: Đảm bảo các class này đã được import đúng từ file tương ứng)
        from initialization import Initialization 
        from variable_neighborhood_search import VariableNeighborhoodSearch
        from energy_efficient_scheduler import EnergyEfficientScheduler
        from rl_agent import RLAgent
        from nsga2_utils import NSGAII_Utils, nextPopulation 
        
        init_module = Initialization(self.pop_size, 0.25, 0.25, 0.25, 0.25, self.jobs, self.factory)
        self.vns = VariableNeighborhoodSearch(self.factory)
        self.es_scheduler = EnergyEfficientScheduler(self.factory)
        self.rl_agent = RLAgent(max_generations=self.max_gen)

        # 2. Population Initialization & Generate Initial Solutions
        population = init_module.generate_population()
        
        # Decode & Evaluate Gen 0
        # (Individual.decode() phiên bản mới sẽ tự tính Makespan, Energy và WCM)
        for ind in population:
            ind.decode()
            
        # Init RL State
        current_state = self.rl_agent.get_state(population, 1)

        # Main Loop
        for gen in range(1, self.max_gen + 1):
            
            # --- 3. RL Agent Select Action (Pc, Pm) ---
            Pc, Pm = self.rl_agent.select_action(current_state, gen)
            
            # --- 4. Perform Crossover & Mutation -> Offspring ---
            offspring = nextPopulation(population, Pc, Pm, self.factory)
            
            # Decode Offspring (Tính toán Fitness & WCM)
            for ind in offspring:
                ind.decode()
            
            # --- 5. RL Learn (Q-Learning / SARSA) ---
            # Conversion Condition Logic (Hình 2):
            # "Whether the conversion conditions are met?" -> No: Q-learning, Yes: SARSA
            # Chiến lược: Dùng Q-Learning ban đầu để khám phá, SARSA về sau để ổn định.
            if gen < self.max_gen * 0.8:
                update_method = 'q_learning'
            else:
                update_method = 'sarsa'
            
            # Gọi hàm update_policy (đã thay thế update_q_table ở class RLAgent)
            self.rl_agent.update_policy(offspring, method=update_method)
            
            # Update State cho vòng sau
            next_state = self.rl_agent.get_state(offspring, gen)
            
            # --- 6. Variable Neighborhood Search (VNS) ---
            combined_pop = population + offspring
            
            if self.vns_enabled:
                # Chạy VNS trên top cá thể tốt nhất của Pareto Front hiện tại
                fronts = NSGAII_Utils.fast_non_dominated_sort(combined_pop)
                top_front = fronts[0]
                
                # Giới hạn chạy trên 5 cá thể tốt nhất để đảm bảo tốc độ
                for i in range(min(5, len(top_front))):
                    original_ind = top_front[i]
                    # Run VNS
                    improved_ind = self.vns.run_vns(original_ind)
                    
                    # Nếu tìm được giải pháp tốt hơn về Makespan
                    if improved_ind.makespan < original_ind.makespan:
                        # Đảm bảo các chỉ số (WCM, Energy) được tính đủ
                        if improved_ind.wcm == 0: improved_ind.decode()
                        combined_pop.append(improved_ind)

            # --- 7. Energy Efficient Strategy (ES) ---
            if self.es_enabled:
                # Lấy lại Pareto Front từ tập hợp mới (bao gồm cả kết quả VNS)
                fronts = NSGAII_Utils.fast_non_dominated_sort(combined_pop)
                pareto_for_es = fronts[0]
                
                # Áp dụng 3 chiến lược ES1, ES2, ES3
                improved_es_list = self.es_scheduler.apply_energy_strategy(
                    pareto_for_es, zz_rate=0.3, xx_rate=0.7
                )
                
                # Đảm bảo tính toán đầy đủ
                for ind in improved_es_list:
                    if ind.wcm == 0: ind.decode()
                
                combined_pop.extend(improved_es_list)

            # --- 8. Selection (NSGA-II) ---
            # Chọn lọc sinh tồn cho thế hệ sau dựa trên Rank & Crowding Distance
            population = NSGAII_Utils.select_survivors(combined_pop, self.pop_size)
            
            # Cập nhật state hiện tại
            current_state = next_state
            
            # Log kết quả
            best_makespan = min(ind.makespan for ind in population)
            print(f"Gen {gen}/{self.max_gen} | Best MS: {best_makespan:.1f} | RL: {update_method} (Pc={Pc:.2f}, Pm={Pm:.2f})")

        # 9. End -> Return Final Schedule
        final_fronts = NSGAII_Utils.fast_non_dominated_sort(population)
        print("=== END ===")
        # Trả về tập Pareto tốt nhất (Rank 0)
        return final_fronts[0]