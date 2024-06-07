import numpy as np

# Define a transition matrix
transition_matrix = np.array([[0.9, 0.1],
                              [0.5, 0.5]])

# Define an initial state vector
initial_state = np.array([1, 0])  # Starting in state 0

# Number of steps to simulate
steps = 10

# Function to simulate the Markov chain
def markov_chain_simulation(P, x, steps):
    state_distributions = [x]
    for _ in range(steps):
        x = np.dot(x, P)
        state_distributions.append(x)
    return state_distributions

# Simulate the Markov chain
state_distributions = markov_chain_simulation(transition_matrix, initial_state, steps)

# Display the result
for step, distribution in enumerate(state_distributions):
    print(f"Step {step}: {distribution}")
