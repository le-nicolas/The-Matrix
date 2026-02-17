from __future__ import annotations

from typing import Sequence

from .matrix import Matrix


class MarkovChain:
    def __init__(
        self,
        transition_matrix: Matrix | Sequence[Sequence[float]],
        states: Sequence[str] | None = None,
        tolerance: float = 1e-9,
    ) -> None:
        self.transition_matrix = (
            transition_matrix
            if isinstance(transition_matrix, Matrix)
            else Matrix(transition_matrix)
        )
        if not self.transition_matrix.is_square:
            raise ValueError("Transition matrix must be square.")

        self.size = self.transition_matrix.nrows
        self.tolerance = tolerance

        if states is None:
            self.states = [f"S{i}" for i in range(self.size)]
        else:
            if len(states) != self.size:
                raise ValueError(
                    "State names length must match matrix size "
                    f"({len(states)} != {self.size})."
                )
            self.states = list(states)

        self._validate_transition_matrix()

    def _validate_transition_matrix(self) -> None:
        for row_index, row in enumerate(self.transition_matrix.to_list()):
            if any(value < -self.tolerance for value in row):
                raise ValueError(
                    f"Transition probabilities must be non-negative (row {row_index})."
                )
            row_sum = sum(row)
            if abs(row_sum - 1.0) > self.tolerance:
                raise ValueError(
                    f"Transition row {row_index} must sum to 1.0 (got {row_sum:.6f})."
                )

    def _validate_distribution(self, distribution: Sequence[float]) -> list[float]:
        values = [float(value) for value in distribution]
        if len(values) != self.size:
            raise ValueError(
                "Distribution length must match matrix size "
                f"({len(values)} != {self.size})."
            )
        if any(value < -self.tolerance for value in values):
            raise ValueError("Distribution values must be non-negative.")
        total = sum(values)
        if abs(total - 1.0) > self.tolerance:
            raise ValueError(f"Distribution must sum to 1.0 (got {total:.6f}).")
        return values

    def step(self, distribution: Sequence[float]) -> list[float]:
        current = self._validate_distribution(distribution)
        return self.transition_matrix.multiply_row_vector(current)

    def simulate(self, initial_distribution: Sequence[float], steps: int) -> list[list[float]]:
        if steps < 0:
            raise ValueError("Steps must be non-negative.")

        current = self._validate_distribution(initial_distribution)
        history = [current]
        for _ in range(steps):
            current = self.step(current)
            history.append(current)
        return history

    def n_step_transition(self, steps: int) -> Matrix:
        if steps < 0:
            raise ValueError("Steps must be non-negative.")
        return self.transition_matrix.power(steps)

    def distribution_after(self, initial_distribution: Sequence[float], steps: int) -> list[float]:
        if steps < 0:
            raise ValueError("Steps must be non-negative.")
        initial = self._validate_distribution(initial_distribution)
        return self.transition_matrix.power(steps).multiply_row_vector(initial)

    def stationary_distribution(
        self, max_iterations: int = 10000, tolerance: float = 1e-12
    ) -> list[float]:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive.")

        current = [1.0 / self.size] * self.size
        for _ in range(max_iterations):
            updated = self.step(current)
            delta = max(abs(a - b) for a, b in zip(current, updated))
            current = updated
            if delta < tolerance:
                return current

        raise RuntimeError("Stationary distribution did not converge.")

    def most_likely_state(self, distribution: Sequence[float]) -> str:
        values = self._validate_distribution(distribution)
        index = max(range(self.size), key=lambda i: values[i])
        return self.states[index]


def format_distribution(distribution: Sequence[float], digits: int = 4) -> str:
    return ", ".join(f"{value:.{digits}f}" for value in distribution)
