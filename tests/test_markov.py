import unittest

from the_matrix import MarkovChain


class MarkovChainTests(unittest.TestCase):
    def test_invalid_transition_matrix_raises(self) -> None:
        with self.assertRaises(ValueError):
            MarkovChain([[0.6, 0.3], [0.2, 0.7]])

    def test_simulate_returns_steps_plus_initial(self) -> None:
        chain = MarkovChain([[0.9, 0.1], [0.5, 0.5]], states=["A", "B"])
        history = chain.simulate([1.0, 0.0], steps=3)
        self.assertEqual(len(history), 4)
        self.assertAlmostEqual(history[1][0], 0.9, places=12)
        self.assertAlmostEqual(history[1][1], 0.1, places=12)

    def test_stationary_distribution(self) -> None:
        chain = MarkovChain([[0.9, 0.1], [0.5, 0.5]], states=["A", "B"])
        stationary = chain.stationary_distribution()
        self.assertAlmostEqual(stationary[0], 5.0 / 6.0, places=5)
        self.assertAlmostEqual(stationary[1], 1.0 / 6.0, places=5)

    def test_distribution_after_matches_simulation(self) -> None:
        chain = MarkovChain([[0.9, 0.1], [0.5, 0.5]], states=["A", "B"])
        direct = chain.distribution_after([1.0, 0.0], steps=5)
        simulated = chain.simulate([1.0, 0.0], steps=5)[-1]
        for a, b in zip(direct, simulated):
            self.assertAlmostEqual(a, b, places=12)

    def test_most_likely_state(self) -> None:
        chain = MarkovChain([[0.9, 0.1], [0.5, 0.5]], states=["Sunny", "Rainy"])
        self.assertEqual(chain.most_likely_state([0.4, 0.6]), "Rainy")


if __name__ == "__main__":
    unittest.main()
