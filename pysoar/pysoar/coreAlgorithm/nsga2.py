
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import itertools
import pandas as pd
import plotly.graph_objects as go
import operator

class Objectives:
    def __init__(self, objectives):
        self.signs = np.ones((len(objectives)))
        for iterate in range(len(objectives)):
            if objectives[iterate]["type"] == "Maximize":
                self.signs[iterate] = -1

        self.objectives = objectives

    def evaluate(self, population):
        m, _ = population.shape
        sol = np.zeros((m, len(self.objectives)))

        for iterate in range(len(self.objectives)):
            sol[:, iterate] = self.signs[iterate] * self.objectives[iterate]["function"](population)
        
        return sol


class Frontiers:
    # class_id = itertools.count(1,1)
    def __init__(self, index):
        self.frontier_id = index
        self.points_in_frontier = []
        

    def add(self, serial_number):
        self.points_in_frontier.append(serial_number)


class FrontiersFactory:
    def __init__(self) -> None:
        self.counter = itertools.count(1,1)
    def create(self):
        index = next(self.counter)
        return Frontiers(index)

class NonDominatedSorting:
    def __init__(self, population):
        self.points_frontier_class = []
        frontier_factory = FrontiersFactory()
        frontier = frontier_factory.create()

        for iterate_1 in range(len(population.population)):
            for iterate_2 in range(len(population.population)):
                # print("********")
                # print(population.population[iterate_1].serial_number)
                # print(population.population[iterate_2].serial_number)
                # print("********")
                if self.dominates(population.population[iterate_1].corres_eval, 
                                    population.population[iterate_2].corres_eval):

                    population.population[iterate_1].S.add(
                                            population.population[iterate_2].serial_number
                                            )
                                           
                elif self.dominates(population.population[iterate_2].corres_eval, 
                                    population.population[iterate_1].corres_eval):
                    
                    population.population[iterate_1].n += 1

            if population.population[iterate_1].n == 0:
                population.population[iterate_1].rank = 1
                frontier.add(population.population[iterate_1].serial_number)
            # print(population.population[iterate_1].S)
        all_frontiers = [frontier]
        iterate = 0
        while not len(all_frontiers[iterate].points_in_frontier)==0:
            Q = []
            for p_id in all_frontiers[iterate].points_in_frontier:
                point_p, index_p = population.fetch_by_serial_number(p_id)
                # print(list([point_p.S]))
                for q_id in list(point_p.S):
                    # print("Hh")
                    # print(q_id)
                    # print([iterate.serial_number for iterate in population.population])
                    point_q, index_q = population.fetch_by_serial_number(q_id)
                    population.population[index_q].n -= 1
                    if population.population[index_q].n == 0:
                        # print(population.population[index_q].rank)
                        population.population[index_q].rank = iterate + 2
                        # print(population.population[index_q].rank)
                        Q.append(q_id)
            
            # for x in all_frontiers[iterate].points_in_frontier:
            #     _, index = population.fetch_by_serial_number(x)
                
            #     population.population[index].frontier = all_frontiers[iterate].frontier_id

            iterate += 1
            # print("Length Q = {}".format(len(Q)))
            
            next_frontier = frontier_factory.create()
            for id in Q:
                next_frontier.add(id)
            
            all_frontiers.append(next_frontier)

        self.all_frontiers = all_frontiers[:-1]

    def crowding_distance(self, population):
        all_eval = np.array([iterate.corres_eval for iterate in population.population])

        for _, frontiers in enumerate(self.all_frontiers):
            cardinality_r = len(frontiers.points_in_frontier)
            evaluations = []
            sr_num = []
            for iterate in frontiers.points_in_frontier:
                point,_ = population.fetch_by_serial_number(iterate)
                evaluations.append(point.corres_eval)
                sr_num.append(point.serial_number)
            evaluations = np.array(evaluations)
            sr_num = np.array(sr_num)
            for objective_num in range(population.num_objectives):
                
                sub_evaluations = evaluations[:, objective_num]
                
                sort_indices = np.argsort(sub_evaluations)
                
                sub_evaluations = sub_evaluations[sort_indices]
                # print(sub_evaluations)
                sr_num = sr_num[sort_indices]

                _, first_index = population.fetch_by_serial_number(sr_num[0])
                _, last_index = population.fetch_by_serial_number(sr_num[-1])

                population.population[first_index].d = float('inf')
                population.population[last_index].d = float('inf')
                low_val = min(all_eval[:, objective_num])
                high_val = max(all_eval[:, objective_num])
                for i in range(1, cardinality_r-1):
                    
                    _, index_mid = population.fetch_by_serial_number(sr_num[i])
                    # index_high_d, _ = population.fetch_by_serial_number(sr_num[i+1])
                    # population.population[index_mid].d += ((
                    #                         index_high_d.d - index_low_d.d
                    #                     )) / (high_val - low_val + 1e-30)
                    
                    population.population[index_mid].d += (abs(
                                            sub_evaluations[i+1] - sub_evaluations[i-1]
                                        )) / (high_val - low_val + 1e-60)
                    # print((evaluations_sub[cardinality_r-1], evaluations_sub[0]))
                # print([iterate.d for iterate in population.population])
        # print(f)
        # print("*******************************")

    def dominates(self, p, q):
        comparison = p < q
        if np.all(comparison):
            return True
        else:
            return False

class SolutionVecProps:
    def __init__(self, sol_vec, corres_eval, index):
        self.serial_number = index
        self.rank = 0
        self.sol_vec = sol_vec
        self.corres_eval = corres_eval
        self.S = set()
        self.n = 0
        self.d = 0
        self.frontier = -1

class SolutionVecPropsFactory:
    def __init__(self) -> None:
        self.counter = itertools.count()
    def create(self, sol_vec, corres_eval):
        index = next(self.counter)
        return SolutionVecProps(sol_vec, corres_eval, index)

class Population:
    def __init__(self, population_size, num_variables, bounds, 
                    objectives, seed, defined_pop = [], generate = True):
        self.seed = seed
        self.rng = default_rng(seed)
        self.population_size = population_size
        self.num_objectives = len(objectives.objectives)
        self.num_variables = num_variables
        self.bounds = bounds
        self.objectives = objectives
        if generate == True:
            sol_vectors = self.generate_random_legal_population()
            # sol_vectors = np.array([[0.913, 2.181],
            #                         [0.599, 2.450],
            #                         [0.139, 1.157],
            #                         [0.867, 1.505],
            #                         [0.885, 1.239],
            #                         [0.658, 2.040],
            #                         [0.788, 2.166],
            #                         [0.342, 0.756]])
            evaluations = self.evaluate_objectives(sol_vectors)
            self.population = self.generate_population(sol_vectors, evaluations)
        else:
            sol_vectors = defined_pop
            evaluations = self.evaluate_objectives(sol_vectors)
            self.population = self.generate_population(sol_vectors, evaluations)


    def get_all_sol_vecs(self):
        return np.array([iterate.sol_vec for iterate in self.population])

    def get_all_evals(self):
        return np.array([iterate.corres_eval for iterate in self.population])

    def get_all_serial_numbers(self):
        return [iterate.serial_number for iterate in self.population]

    def generate_population(self, sol_vectors, evaluations):
        population = []
        solVecProp = SolutionVecPropsFactory()
        for sol_vec, corres_eval in zip(sol_vectors, evaluations):
            pointProp = solVecProp.create(sol_vec, corres_eval)
            population.append(pointProp)
        return population

    def fetch_by_serial_number(self, target):
        for iterate, pop in enumerate(self.population):
            if pop.serial_number == target:
                return pop, iterate
        

    def generate_random_legal_population(self):
        population = self.rng.random((self.population_size, self.num_variables))
        for iterate in range(self.num_variables):
            lower_b = self.bounds[iterate][0]
            upper_b = self.bounds[iterate][1]
            population[:, iterate] = (population[:, iterate] * (upper_b - lower_b)) + lower_b

        return population

    def plotPopulation(self):
        fig = go.Figure()
        evaluations = self.get_all_evals()
        point_caption = (["Point {}".format(i) for i in self.get_all_serial_numbers()])
        fig.add_trace(go.Scatter(
            x = 1*evaluations[:,0],
            y = 1*evaluations[:,1],
            mode = "markers",
            text = point_caption
        ))
        fig.update_layout(
            width = 800,
            height = 800,
            title = "fixed-ratio axes"
        )
        fig.update_yaxes(
            range = [-12, 2],
            scaleanchor = "x",
            scaleratio = 1,
        )
        fig.update_xaxes(
            range = [-20,-14],
            scaleanchor = "x",
            scaleratio = 1,
        )
        fig.show()
    
    def plotPopulationwithFrontier(self):
        fig = go.Figure()
        all_frontiers = NonDominatedSorting(self)
        print(len(all_frontiers.all_frontiers))
        for rank, frontiers in enumerate(all_frontiers.all_frontiers):
            evaluations = []
            sr_num = []
            for iterate in frontiers.points_in_frontier:
                point,_ = self.fetch_by_serial_number(iterate)
                evaluations.append(point.corres_eval)
                sr_num.append(point.serial_number)
            evaluations = np.array(evaluations)
            df = pd.DataFrame(dict(
                    x = 1*evaluations[:,0],
                    y = 1*evaluations[:,1],
                ))
            print(df)
            point_caption = (["Point {}".format(i) for i in sr_num])
            fig.add_trace(go.Scatter(
            x = df.sort_values(by="x")["x"],
            y = df.sort_values(by="x")["y"],
            mode = "markers+lines",
            text = point_caption,
            name = "Frontier {}".format(rank + 1)
            ))

        fig.update_layout(
            width = 800,
            height = 800,
            title = "fixed-ratio axes"
        )
        fig.update_yaxes(
            range = [-12, 2],
            scaleanchor = "x",
            scaleratio = 1,
        )
        fig.update_xaxes(
            range = [-20,-14],
            scaleanchor = "x",
            scaleratio = 1,
        )
        fig.show()

    def evaluate_objectives(self, sol_vectors):
        # population = self.population
        return self.objectives.evaluate(sol_vectors)

    def thanos_kill_move(self):
        
        ranks = [iterate.rank for iterate in self.population]
        cd = [-1*iterate.d for iterate in self.population]
        serial_num = [iterate.serial_number for iterate in self.population]
        combined_array = np.array([ranks, cd, serial_num]).T.tolist()
        combined_array = np.array(sorted(combined_array, key = operator.itemgetter(0,1)))
        # print(combined_array)
        # print(f)
        selected_indices = combined_array[0:self.population_size,-1].astype(int)

        survivors = []
        for iterate in selected_indices:
            point, _ = self.fetch_by_serial_number(iterate)
            
            survivors.append(point.sol_vec)
            
        
        survivors = np.array(survivors)
        # for survivor in survivors:
            
        #     for iterate,j in enumerate(self.bounds):
        #         if survivor[iterate] < j[0]:
        #             survivors[iterate][0] = j[0]
        #         if survivor[iterate] > j[1]:
        #             survivors[iterate][1] = j[1]
        # print(survivors)
        # print(f)
        
        return Population(self.population_size, self.num_variables,
                            self.bounds, self.objectives, self.seed+3,
                            survivors, False)


    def find_bounds(self, ranks, target):
        lower_bound = self.binarySearch(ranks, target,True)
        upper_bound = self.binarySearch(ranks, target,False, lower_bound)
        return lower_bound, upper_bound
    
    def find_upper_bound(self, ranks, target):
        upper_bound = self.binarySearch(ranks, target,False)
        return upper_bound

    def binarySearch(self, inp, target, lowerBound, start = 0):
        left = start
        right = len(inp) - 1

        while left <= right:
            mid = (left + right) // 2
            # print(left, mid, right)
            # print(inp[left], inp[mid], inp[right])
            if inp[mid] == target:
                if inp[mid] == target:
                    if lowerBound:
                        if mid == 0:
                            return mid
                        elif inp[mid-1] != target:
                            return mid
                        else:
                            right = mid
                    else:
                        if mid == len(inp)-1:
                            return mid
                        elif inp[mid+1] != target:
                            return mid
                        else:
                            left = mid + 1

            elif inp[mid] < target:
                left = mid + 1

            else:
                right = mid - 1

    
class GARoutine:
    def __init__(self, population, seed) -> None:
        self.rng = default_rng(seed)

        self.sol_vec = population.get_all_sol_vecs()
        self.crowding_distance = np.array(
                            [iterate.d for iterate in population.population]
                        )
        self.rank = np.array(
                            [iterate.rank for iterate in population.population]
                        )
        self.size = len(population.population)
        self.bounds = population.bounds
        # print(self.sol_vec)
        # print(self.crowding_distance)
        # print(self.rank)

    def crowded_binary_tournament_selection(self, withReplacement = False):
        # print(self.size)
        size = self.size
        chosen = []
        for _ in range(2):
            
            tournament_draw = self.rng.choice(size, size = size, replace = withReplacement)
            # print(tournament_draw)
            if size % 2 == 0:
                for iterate in range(0, size, 2):
                    chosen.append(self.choose(tournament_draw[iterate], tournament_draw[iterate+1]))
                
            else:
                raise ValueError("Population Size should be Even integer.")
        # print(chosen)
        winners = []
        for i in chosen:
            winners.append(self.sol_vec[i, :])

        return np.array(winners)

    def choose(self, parent_1_index, parent_2_index):
        
        if self.rank[parent_1_index] != self.rank[parent_2_index]:
            rank_p1 = self.rank[parent_1_index]
            rank_p2 = self.rank[parent_2_index]
            return parent_1_index if rank_p1 < rank_p2 else parent_2_index
        elif self.crowding_distance[parent_1_index] != self.crowding_distance[parent_2_index]:
            cd_p1 = self.crowding_distance[parent_1_index]
            cd_p2 = self.crowding_distance[parent_2_index]
            return parent_1_index if cd_p1 > cd_p2 else parent_2_index
        else:
            toss = self.rng.random()
            return parent_1_index if toss < 0.5 else parent_2_index

    def sbx_crossover_operator(self, sol_vec, crossover_prob, p_curve_param, withReplacement = False):
        crossover_couples = self.rng.choice(self.size, size = self.size, replace = withReplacement)
        offsprings = []
        
        if self.size%2 == 0:
            for iterate in range(0, self.size, 2):
                offspring_1, offspring_2 = self.generate_offspring_from_SBX(sol_vec, crossover_couples[iterate],
                                                    crossover_couples[iterate + 1],
                                                    p_curve_param, crossover_prob)
                offsprings.append(offspring_1)
                offsprings.append(offspring_2)                                
        else: 
            raise ValueError("Population Size should be Even integer.")
        
        return np.array(offsprings)
                                                                                                                 

    def generate_offspring_from_SBX(self, sol_vec, p1_index, p2_index, p_curve_param, crossover_prob):
        biased_toss = self.rng.random()
        if biased_toss <= crossover_prob:
            beta = self.calculate_beta(p_curve_param, biased_toss)
            # print("Crossover Took Place: {}, {}".format(biased_toss, crossover_prob))
            p1 = sol_vec[p1_index,:]
            p2 = sol_vec[p2_index, :]
            child_1 = 0.5 * ((p1 + p2) + beta*(p1-p2))
            child_2 = 0.5 * ((p1 + p2) - beta*(p1-p2))
            # print(child_1)
            # print(child_2)
            # print(f)

            offsprings_1 = child_1
            offsprings_2 = child_2

            for iterate in range(len(self.bounds)):
                
                if offsprings_1[iterate] < self.bounds[iterate][0] or offsprings_1[iterate] > self.bounds[iterate][1]:
                    
                    offsprings_1[iterate] = self.bounds[iterate][0] + self.rng.random() * (self.bounds[iterate][1]-self.bounds[iterate][0])
            
            
            
            for iterate in range(len(self.bounds)):
                
                if offsprings_2[iterate] < self.bounds[iterate][0] or offsprings_2[iterate] > self.bounds[iterate][1]:
                    
                    offsprings_2[iterate] = self.bounds[iterate][0] + self.rng.random() * (self.bounds[iterate][1]-self.bounds[iterate][0])
            
        else:
            offsprings_1 = sol_vec[p1_index,:]
            offsprings_2 = sol_vec[p2_index,:]

        return offsprings_1, offsprings_2

    def calculate_beta(self, p_curve_param, toss):
        # toss = self.rng.random()
        if toss <= 0.5:
            beta = (2*toss)**(1/(p_curve_param + 1))
        else: 
            beta = (1/(2*(1-toss)))**(1/(p_curve_param + 1)) 
        
        return beta

    def polynomial_mutation_operator(self, sol_vec, bounds, mutation_prob, p_curve_param_mutation):
        offsprings = []

        bound_length = []
        for b in bounds:
            bound_length.append(b[1] - b[0])
        bound_length = np.array(bound_length)
        
# future edit needed here
        for iterate in range(self.size):
            biased_toss = self.rng.random()
            
            if biased_toss <= mutation_prob:
                delta_bar = self.calculate_delta_bar(p_curve_param_mutation, biased_toss)
                # mut_offspring = []
                # print(bound_length)
                mut_offspring = sol_vec[iterate,:] + ((bound_length) * delta_bar)
                for iterate in range(len(self.bounds)):
                
                    if mut_offspring[iterate] < self.bounds[iterate][0] or mut_offspring[iterate] > self.bounds[iterate][1]:
                        
                        mut_offspring[iterate] = self.bounds[iterate][0] + self.rng.random() * (self.bounds[iterate][1]-self.bounds[iterate][0])
                # print(f)
                # for iterate_2, b in bounds:
                    
                #     if gen < b[]
                #     mut_offspring = 

                offsprings.append(mut_offspring)
            else:
                offsprings.append(sol_vec[iterate,:])
        

        return np.array(offsprings)
                                                                                                                 

    def calculate_delta_bar(self, p_curve_param, toss):
        # toss = self.rng.random()
        if toss <= 0.5:
            delta_bar = ((2*toss)**(1/(p_curve_param + 1))) - 1
        else: 
            delta_bar = 1 - ((2*(1-toss))**(1/(p_curve_param + 1))) 
        
        return delta_bar
