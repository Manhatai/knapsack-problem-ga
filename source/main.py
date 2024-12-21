from knapsack.knapsack import KnapsackProblemFinal
from knapsack.knapsack_testing import KnapsackProblemTest
from settings import __knapsack_config, __test_config, IS_DEBUG

if IS_DEBUG == 'False':
    KnapsackProblemFinal(
        csv_row=__knapsack_config["csv_row"],
        vector_length=__knapsack_config["vector_length"],
        chromosome_count=__knapsack_config["chromosome_count"],
        relative_threshold_value=__knapsack_config["relative_threshold_value"],
        tournament_size=__knapsack_config["tournament_size"],
        mutation_rate=__knapsack_config["mutation_rate"],
        generation_count=__knapsack_config["generation_count"]
    ).run()
else:
    KnapsackProblemTest(
        csv_row=__test_config["csv_row"],
        vector_length=__test_config["vector_length"],
        chromosome_count=__test_config["chromosome_count"],
        relative_threshold_value=__test_config["relative_threshold_value"],
        tournament_size=__test_config["tournament_size"],
        mutation_rate=__test_config["mutation_rate"],
    ).run()