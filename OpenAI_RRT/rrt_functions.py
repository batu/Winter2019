import random


# Embedding Functions
def no_embedding(input: "numpy_array") -> "numpy_array":
    return input


# Action selection functions
def random_action(action_space: int) -> int:
    return random.randint(0, action_space.n)
