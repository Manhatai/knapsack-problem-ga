import pandas as pd
import numpy as np
from pandas import DataFrame
import random

class KnapsackProblemTest:
    def __init__(self, csv_row, vector_length, chromosome_count, relative_threshold_value, tournament_size, mutation_rate):
        self.csv_row = csv_row
        self.vector_length = vector_length
        self.chromosome_count = chromosome_count
        self.relative_threshold_value = relative_threshold_value
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate

    def run(self) -> None:
        df = pd.read_csv("data/zbior_danych_ag.csv")
        self.__process_input_data(df)

    def __process_input_data(self, df: DataFrame) -> None:
        raw_csv_string = df.iloc[self.csv_row, 0]
        parts_of_raw_csv_string = raw_csv_string.split(';') # ['[46 40 42 38 10]', '[12 19 19 15  8]', '40']
        ciezar = list(map(int, parts_of_raw_csv_string[0].strip('[]').split())) # '46 40 42 38 10', ['46', '40', '42', '38', '10'], [46, 40, 42, 38, 10]
        ceny = list(map(int, parts_of_raw_csv_string[1].strip('[]').split()))
        pojemnosc = int(parts_of_raw_csv_string[2])                              
        self.__output_csv(ciezar, ceny, pojemnosc)
        
        random_chromosomes = self.__random_chromosome_generator()
        valid_chromosomes = self.__fitness_function(random_chromosomes, ciezar, ceny, pojemnosc)
        self.__test_selection_operators(valid_chromosomes, ciezar, ceny, pojemnosc, tournament_size=self.tournament_size)
        self.__test_crossover_operators(valid_chromosomes, ciezar, ceny, pojemnosc)
        self.__test_mutation_operator(valid_chromosomes, mutation_rate=self.mutation_rate)

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

    def __fitness_function(self, random_chromosomes: list[list[int]], ciezar: list[int], ceny: list[int], pojemnosc: int) -> list[list[int]]:
        valid_chromosomes = []
        max_total_value = sum(ceny)
        relative_threshold = self.relative_threshold_value * max_total_value
        for chromosome in random_chromosomes:
            total_weight = sum(chromosome_bool * weight for chromosome_bool, weight in zip(chromosome, ciezar))
            total_value = sum(chromosome_bool * value for chromosome_bool, value in zip(chromosome, ceny))
            is_valid = total_weight <= pojemnosc
            if total_weight > pojemnosc:
                penalty = max(0, (1 - ((total_weight - pojemnosc) / pojemnosc)))
                total_value *= penalty
            print(f"Chromosom: {chromosome}, Ciężar: {total_weight}, Cena po odjęciu kary: {total_value:.2f}, Poprawny: {is_valid}")
            if total_value >= relative_threshold and is_valid:
                valid_chromosomes.append(chromosome)
        print(f"Ilość chromosomów do dalszego krzyżowania: {len(valid_chromosomes)}")
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
        print(f"\nPunkt krzyżowania dla jednopunktowego to: {crossover_point}")
        return child1, child2

    def __crossover_double_point(self, parent1: list[int], parent2: list[int]) -> tuple[list[int], list[int]]:
        crossover_point1 = random.randint(1, len(parent1) - 2) # -2 = no overlap
        crossover_point2 = random.randint(crossover_point1 + 1, len(parent1) - 1)
        child1 = parent1[:crossover_point1] + parent2[crossover_point1:crossover_point2] + parent1[crossover_point2:]
        child2 = parent2[:crossover_point1] + parent1[crossover_point1:crossover_point2] + parent2[crossover_point2:]
        print(f"\nPunkty krzyżowania dla dwupunktowego to: {crossover_point1} i {crossover_point2}")
        return child1, child2

    def __tournament_selection(self, valid_chromosomes: list[list[int]], ciezar: list[int], ceny: list[int], pojemnosc: int, tournament_size: int) -> list[int]:
        if len(valid_chromosomes) < tournament_size:
            print("Za mało chromosomów do turnieju!")
            return random.choice(valid_chromosomes)
        tournament = random.sample(valid_chromosomes, tournament_size)
        fitness_scores = [self.__fitness_function_single(chromosome, ciezar, ceny, pojemnosc) for chromosome in tournament]
        best_index = np.argmax(fitness_scores)
        print(f"Turniej: {tournament}")
        return tournament[best_index]

    def __roulette_wheel_selection(self, valid_chromosomes: list[list[int]], ciezar: list[int], ceny: list[int], pojemnosc: int) -> list[int]:
        fitness_scores = [self.__fitness_function_single(chromosome, ciezar, ceny, pojemnosc) for chromosome in valid_chromosomes]
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            print("Brak różnic w funkcji, wybór losowy.")
            return random.choice(valid_chromosomes)
        probabilities = [(fitness / total_fitness) for fitness in fitness_scores]
        selected_index = np.random.choice(len(valid_chromosomes), p=probabilities)
        print(f"Prawdopodobieństwa: {probabilities}, Wybrany: {valid_chromosomes[selected_index]}")
        return valid_chromosomes[selected_index]
    
    def __point_mutation(self, chromosome: list[int], mutation_rate: float) -> list[int]:
        mutated_chromosome = chromosome[:]
        for i in range(len(mutated_chromosome)):
            if random.random() < mutation_rate:
                mutated_chromosome[i] = 1 - mutated_chromosome[i]
        return mutated_chromosome

    def __test_crossover_operators(self, valid_chromosomes: list[list[int]], ciezar: list[int], ceny: list[int], pojemnosc: int) -> None:
        if len(valid_chromosomes) < 2:
            print("Za mało chromosomów do krzyżowania!")
            return
        parent1, parent2 = random.sample(valid_chromosomes, 2)
        print(f"\nRodzice:\nRodzic1: {parent1} Rodzic2: {parent2}")

        child1, child2 = self.__crossover_single_point(parent1, parent2)
        print(f"\nJednopunktowe krzyżowanie:\nDziecko1: {child1}, Dziecko2: {child2}")
        print("Wynik funkcji fitness dla powstałych dzieci:")
        self.__fitness_function([child1, child2], ciezar, ceny, pojemnosc)

        child1, child2 = self.__crossover_double_point(parent1, parent2)
        print(f"\nDwupunktowe krzyżowanie:\nDziecko1: {child1}, Dziecko2: {child2}")
        print("Wynik funkcji fitness dla powstałych dzieci:")
        self.__fitness_function([child1, child2], ciezar, ceny, pojemnosc)
        
    def __test_selection_operators(self, valid_chromosomes: list[list[int]], ciezar: list[int], ceny: list[int], pojemnosc: int, tournament_size: int) -> None:
        if len(valid_chromosomes) < 2:
            print("Za mało chromosomów do testowania selekcji!")
            return

        print("\nTest selekcji turniejowej:")
        best_tournament = self.__tournament_selection(valid_chromosomes, ciezar, ceny, pojemnosc, tournament_size)
        print(f"Najlepszy z turnieju: {best_tournament}")

        print("\nTest selekcji proporcjonalnej:")
        best_roulette = self.__roulette_wheel_selection(valid_chromosomes, ciezar, ceny, pojemnosc)
        print(f"Wybrany przez ruletkę: {best_roulette}")
        
    def __test_mutation_operator(self, valid_chromosomes: list[list[int]], mutation_rate: float) -> None:
        print("\nTest operatora mutacji punktowej:")
        for chromosome in valid_chromosomes:
            mutated = self.__point_mutation(chromosome, mutation_rate)
            print(f"Chromosom przed mutacją: {chromosome}, Chromosom po mutacji: {mutated}")