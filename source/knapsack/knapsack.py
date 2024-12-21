import pandas as pd
import numpy as np
from pandas import DataFrame
import random

class KnapsackProblemFinal:
    def __init__(self, csv_row, vector_length, chromosome_count, relative_threshold_value, tournament_size, mutation_rate, generation_count):
        self.csv_row = csv_row
        self.vector_length = vector_length
        self.chromosome_count = chromosome_count
        self.relative_threshold_value = relative_threshold_value
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.generation_count = generation_count

    def run(self) -> None:
        df = pd.read_csv("./knapsack/data/zbior_danych_ag.csv")
        self.__process_input_data(df)

    def __process_input_data(self, df: DataFrame) -> None:
        raw_csv_string = df.iloc[self.csv_row, 0]
        parts_of_raw_csv_string = raw_csv_string.split(';')
        ciezar = list(map(int, parts_of_raw_csv_string[0].strip('[]').split()))
        ceny = list(map(int, parts_of_raw_csv_string[1].strip('[]').split()))
        pojemnosc = int(parts_of_raw_csv_string[2])
        self.__output_csv(ciezar, ceny, pojemnosc)
        self.__run_generations(ciezar, ceny, pojemnosc)
        

    def __output_csv(self, ciezar: list[int], ceny: list[int], pojemnosc: int) -> None:
        output_data = {
            "Ciezar": ciezar,
            "Ceny": ceny,
            "Pojemnosc": pojemnosc
        }
        print(f"\nAktualnie używane dane: {output_data}\n")

    def __random_chromosome_generator(self) -> list[list[int]]:
        random_chromosomes = []
        for _ in range(self.chromosome_count): 
            binary_vector = [random.randint(0, 1) for _ in range(self.vector_length)]
            random_chromosomes.append(binary_vector)
        return random_chromosomes
    
    def __run_generations(self, ciezar: list[int], ceny: list[int], pojemnosc: int) -> None:
        population = self.__random_chromosome_generator()
        for generation in range(self.generation_count):
            print(f"\nGeneracja {generation + 1}:")
            valid_chromosomes = self.__fitness_function(population, ciezar, ceny, pojemnosc)
            stats = self.__calculate_population_statistics(valid_chromosomes, ciezar, ceny, pojemnosc)
            print(f"Najlepsza cena: {stats['best']}, Najgorsza cena: {stats['worst']}, Średnia cena: {stats['average']}")
            next_generation = []
            while len(next_generation) < self.chromosome_count:
                parent1 = self.__tournament_selection(valid_chromosomes, ciezar, ceny, pojemnosc, self.tournament_size)
                parent2 = self.__tournament_selection(valid_chromosomes, ciezar, ceny, pojemnosc, self.tournament_size)
                child1, child2 = self.__crossover_single_point(parent1, parent2)
                next_generation.extend([child1, child2])
            population = [self.__point_mutation(chromosome, self.mutation_rate) for chromosome in next_generation]

    def __fitness_function(self, population: list[list[int]], ciezar: list[int], ceny: list[int], pojemnosc: int) -> list[list[int]]:
        valid_chromosomes = []
        max_total_value = sum(ceny)
        relative_threshold = self.relative_threshold_value * max_total_value
        for chromosome in population:
            total_weight = sum(chromosome_bool * weight for chromosome_bool, weight in zip(chromosome, ciezar))
            total_value = sum(chromosome_bool * value for chromosome_bool, value in zip(chromosome, ceny))
            if total_weight > pojemnosc:
                penalty = max(0, (1 - ((total_weight - pojemnosc) / pojemnosc)))
                total_value *= penalty
            is_valid = total_weight <= pojemnosc
            #print(f"Chromosom: {chromosome}, Ciężar: {total_weight}, Cena: {total_value:.2f}, Poprawny: {is_valid}")
            if total_value >= relative_threshold and is_valid:
                valid_chromosomes.append(chromosome)
        chromosome_count = len(valid_chromosomes)
        if chromosome_count <= 1:
            print(f"Niewystarczająca liczba chromosomów aby kontynuować! ({chromosome_count})")
            exit()
        print(f"Ilość chromosomów do dalszego krzyżowania: {chromosome_count}")
        return valid_chromosomes

    def __fitness_function_single(self, chromosome: list[int], ciezar: list[int], ceny: list[int], pojemnosc: int) -> float:
        total_weight = sum(chromosome_bool * weight for chromosome_bool, weight in zip(chromosome, ciezar))
        total_value = sum(chromosome_bool * value for chromosome_bool, value in zip(chromosome, ceny))
        if total_weight > pojemnosc:
            penalty = max(0, (1 - ((total_weight - pojemnosc) / pojemnosc)))
            total_value *= penalty
        return total_value

    def __crossover_single_point(self, parent1: list[int], parent2: list[int]) -> tuple[list[int], list[int]]:
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def __tournament_selection(self, valid_chromosomes: list[list[int]], ciezar: list[int], ceny: list[int], pojemnosc: int, tournament_size: int) -> list[int]:
        if len(valid_chromosomes) < tournament_size:
            print("Za mało chromosomów do turnieju!")
            return random.choice(valid_chromosomes)
        tournament = random.sample(valid_chromosomes, tournament_size)
        fitness_scores = [self.__fitness_function_single(chromosome, ciezar, ceny, pojemnosc) for chromosome in tournament]
        best_index = np.argmax(fitness_scores)
        return tournament[best_index]

    def __point_mutation(self, chromosome: list[int], mutation_rate: float) -> list[int]:
        mutated_chromosome = chromosome[:]
        for i in range(len(mutated_chromosome)):
            if random.random() < mutation_rate:
                mutated_chromosome[i] = 1 - mutated_chromosome[i]
        return mutated_chromosome

    def __calculate_population_statistics(self, population: list[list[int]], ciezar: list[int], ceny: list[int], pojemnosc: int) -> dict:
        fitness_scores = [self.__fitness_function_single(chromosome, ciezar, ceny, pojemnosc) for chromosome in population]
        stats = {
            "best": max(fitness_scores),
            "worst": min(fitness_scores),
            "average": (sum(fitness_scores) / len(fitness_scores))
        }
        return stats