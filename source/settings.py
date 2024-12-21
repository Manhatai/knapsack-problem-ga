import os
from dotenv import load_dotenv
load_dotenv()

__knapsack_config = {
    "csv_row": 2,
    "vector_length": 5,    
    "chromosome_count": 100,
    "relative_threshold_value": 0.6,
    "tournament_size": 2,
    "mutation_rate": 0.08,
    "generation_count": 5,
}

__test_config = {
    "csv_row": 2,
    "vector_length": 5,    
    "chromosome_count": 100,
    "relative_threshold_value": 0.6,
    "tournament_size": 3,
    "mutation_rate": 0.2,
}

IS_DEBUG = os.environ.get('IS_DEBUG')