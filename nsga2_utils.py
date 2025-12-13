import random
import copy

class NSGAII_Utils:
    """
    Các hàm hỗ trợ sắp xếp không trội (Non-dominated Sorting) và Crowding Distance
    cho bài toán đa mục tiêu (MCT, TEC, WCM).
    """
    
    @staticmethod
    def dominates(ind1, ind2):
        """
        Kiểm tra xem ind1 có trội hơn ind2 không.
        Mục tiêu: Minimization cho cả 3 (MCT, TEC, WCM).
        """
        # Đảm bảo đã decode để có chỉ số
        obj1 = [ind1.makespan, ind1.total_energy, ind1.wcm]
        obj2 = [ind2.makespan, ind2.total_energy, ind2.wcm]
        
        better_in_at_least_one = False
        for a, b in zip(obj1, obj2):
            if a > b: # Nếu ind1 tệ hơn ở bất kỳ mục tiêu nào -> Không trội
                return False
            if a < b:
                better_in_at_least_one = True
        return better_in_at_least_one

    @staticmethod
    def fast_non_dominated_sort(population):
        """Phân lớp Pareto (Fronts)."""
        fronts = [[]]
        for p in population:
            p.domination_count = 0
            p.dominated_solutions = []
            for q in population:
                if NSGAII_Utils.dominates(p, q):
                    p.dominated_solutions.append(q)
                elif NSGAII_Utils.dominates(q, p):
                    p.domination_count += 1
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)
        
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        
        return fronts[:-1] # Loại bỏ front rỗng cuối cùng

    @staticmethod
    def calculate_crowding_distance(front):
        """Tính khoảng cách đám đông để duy trì đa dạng."""
        l = len(front)
        if l == 0: return
        
        for ind in front:
            ind.crowding_distance = 0
            
        # Với mỗi mục tiêu (0: MCT, 1: TEC, 2: WCM)
        for m in range(3):
            # Sort theo mục tiêu m
            front.sort(key=lambda x: [x.makespan, x.total_energy, x.wcm][m])
            
            # Gán vô cùng cho biên
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Lấy range của mục tiêu
            obj_values = [[x.makespan, x.total_energy, x.wcm][m] for x in front]
            scale = obj_values[-1] - obj_values[0]
            if scale == 0: scale = 1.0
            
            for i in range(1, l - 1):
                dist = (obj_values[i+1] - obj_values[i-1]) / scale
                front[i].crowding_distance += dist

    @staticmethod
    def select_survivors(population, n_survivors):
        """Chọn N cá thể tốt nhất dựa trên Rank và Crowding Distance."""
        fronts = NSGAII_Utils.fast_non_dominated_sort(population)
        next_gen = []
        
        for front in fronts:
            NSGAII_Utils.calculate_crowding_distance(front)
            # Nếu thêm cả front này mà chưa vượt quá N -> Thêm hết
            if len(next_gen) + len(front) <= n_survivors:
                next_gen.extend(front)
            else:
                # Nếu vượt quá -> Sort theo Crowding Distance giảm dần và lấy đủ
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                needed = n_survivors - len(next_gen)
                next_gen.extend(front[:needed])
                break
                
        return next_gen

def nextPopulation(current_pop, Pc, Pm, factory):
    """
    Tạo thế hệ con dựa trên Pc và Pm từ RL Agent.
    Input: Quần thể hiện tại, Xác suất lai ghép (Pc), Xác suất đột biến (Pm).
    Output: Quần thể con (Offspring).
    """
    offspring = []
    pop_size = len(current_pop)
    
    # 1. Tournament Selection để chọn cha mẹ
    pool = []
    for _ in range(pop_size):
        # Binary Tournament
        cand1 = random.choice(current_pop)
        cand2 = random.choice(current_pop)
        
        # So sánh dựa trên Rank (nếu chưa có rank thì decode & sort trước)
        if not hasattr(cand1, 'rank'): cand1.rank = 0 
        if not hasattr(cand2, 'rank'): cand2.rank = 0
        
        if cand1.rank < cand2.rank:
            pool.append(cand1)
        elif cand2.rank < cand1.rank:
            pool.append(cand2)
        else: # Rank bằng nhau thì xét Crowding Distance hoặc Random
            pool.append(random.choice([cand1, cand2]))

    # 2. Crossover & Mutation
    for i in range(0, pop_size, 2):
        parent1 = pool[i]
        # Xử lý lẻ
        if i + 1 >= pop_size:
            offspring.append(copy.deepcopy(parent1))
            break
        parent2 = pool[i+1]
        
        child1, child2 = None, None
        
        # --- Crossover (Dựa trên Pc) ---
        if random.random() < Pc:
            # Lai ghép MS
            c1, c2 = parent1.crossover_machine_selection(parent2)
            # Lai ghép OS (tiếp tục trên kết quả MS)
            child1, child2 = c1.crossover_operation_sequence(c2)
        else:
            child1 = copy.deepcopy(parent1)
            child2 = copy.deepcopy(parent2)
            
        # --- Mutation (Dựa trên Pm) ---
        # Mutation MS
        child1.mutation_machine_selection(mutation_rate=Pm)
        child2.mutation_machine_selection(mutation_rate=Pm)
        
        # Mutation OS
        child1.mutation_operation_sequence(mutation_rate=Pm)
        child2.mutation_operation_sequence(mutation_rate=Pm)
        
        offspring.extend([child1, child2])
        
    return offspring