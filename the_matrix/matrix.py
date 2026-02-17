from __future__ import annotations

from typing import Sequence


class Matrix:
    def __init__(self, rows: Sequence[Sequence[float]]) -> None:
        if not rows:
            raise ValueError("Matrix must have at least one row.")
        width = len(rows[0])
        if width == 0:
            raise ValueError("Matrix rows must have at least one value.")

        normalized: list[list[float]] = []
        for row in rows:
            if len(row) != width:
                raise ValueError("All rows must have the same length.")
            normalized.append([float(value) for value in row])

        self._rows = tuple(tuple(row) for row in normalized)

    @property
    def nrows(self) -> int:
        return len(self._rows)

    @property
    def ncols(self) -> int:
        return len(self._rows[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nrows, self.ncols)

    @property
    def is_square(self) -> bool:
        return self.nrows == self.ncols

    def to_list(self) -> list[list[float]]:
        return [list(row) for row in self._rows]

    def transpose(self) -> Matrix:
        transposed = [
            [self._rows[row_index][column_index] for row_index in range(self.nrows)]
            for column_index in range(self.ncols)
        ]
        return Matrix(transposed)

    def multiply_row_vector(self, vector: Sequence[float]) -> list[float]:
        if len(vector) != self.nrows:
            raise ValueError(
                "Row-vector length must match the matrix row count "
                f"({len(vector)} != {self.nrows})."
            )

        result = [0.0] * self.ncols
        for column_index in range(self.ncols):
            total = 0.0
            for row_index in range(self.nrows):
                total += float(vector[row_index]) * self._rows[row_index][column_index]
            result[column_index] = total
        return result

    def almost_equal(self, other: Matrix, tolerance: float = 1e-9) -> bool:
        if self.shape != other.shape:
            return False
        for i in range(self.nrows):
            for j in range(self.ncols):
                if abs(self._rows[i][j] - other._rows[i][j]) > tolerance:
                    return False
        return True

    def power(self, exponent: int) -> Matrix:
        if exponent < 0:
            raise ValueError("Exponent must be non-negative.")
        if not self.is_square:
            raise ValueError("Matrix power requires a square matrix.")

        result = Matrix.identity(self.nrows)
        base = self
        remaining = exponent

        while remaining > 0:
            if remaining % 2 == 1:
                result = result @ base
            base = base @ base
            remaining //= 2

        return result

    @staticmethod
    def identity(size: int) -> Matrix:
        if size <= 0:
            raise ValueError("Identity matrix size must be positive.")
        rows = [[0.0] * size for _ in range(size)]
        for i in range(size):
            rows[i][i] = 1.0
        return Matrix(rows)

    def __matmul__(self, other: Matrix) -> Matrix:
        if self.ncols != other.nrows:
            raise ValueError(
                "Incompatible dimensions for matrix multiplication "
                f"({self.shape} and {other.shape})."
            )

        product: list[list[float]] = []
        for i in range(self.nrows):
            row: list[float] = []
            for j in range(other.ncols):
                total = 0.0
                for k in range(self.ncols):
                    total += self._rows[i][k] * other._rows[k][j]
                row.append(total)
            product.append(row)
        return Matrix(product)

    def __pow__(self, exponent: int) -> Matrix:
        return self.power(exponent)

    def __getitem__(self, index: int) -> tuple[float, ...]:
        return self._rows[index]

    def __repr__(self) -> str:
        return f"Matrix({self.to_list()!r})"
