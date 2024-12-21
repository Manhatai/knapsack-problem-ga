from knapsack import KnapsackProblem

__config = {
    "csv_row": 2,
    "vector_length": 5,    
    "chromosome_count": 100,
    "relative_threshold_value": 0.5,
    "tournament_size": 3,
    "mutation_rate": 0.2,
}

KnapsackProblem(
    csv_row=__config["csv_row"],
    vector_length=__config["vector_length"],
    chromosome_count=__config["chromosome_count"],
    relative_threshold_value=__config["relative_threshold_value"],
    tournament_size=__config["tournament_size"],
    mutation_rate=__config["mutation_rate"]
).run()