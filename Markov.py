from the_matrix import MarkovChain, Matrix, format_distribution


def main() -> None:
    transitions = Matrix(
        [
            [0.70, 0.20, 0.10],  # Sunny -> (Sunny, Cloudy, Rainy)
            [0.30, 0.40, 0.30],  # Cloudy -> (...)
            [0.20, 0.50, 0.30],  # Rainy -> (...)
        ]
    )
    states = ["Sunny", "Cloudy", "Rainy"]
    chain = MarkovChain(transitions, states=states)

    initial_distribution = [1.0, 0.0, 0.0]
    timeline = chain.simulate(initial_distribution, steps=12)

    print("Weather Markov Chain Simulation")
    print("-" * 32)
    for step, distribution in enumerate(timeline):
        likely = chain.most_likely_state(distribution)
        print(f"Step {step:>2}: [{format_distribution(distribution, digits=3)}]  likely={likely}")

    stationary = chain.stationary_distribution()
    print("\nEstimated stationary distribution")
    print("-" * 32)
    for state, probability in zip(chain.states, stationary):
        print(f"{state:<7} {probability:.4f}")


if __name__ == "__main__":
    main()
