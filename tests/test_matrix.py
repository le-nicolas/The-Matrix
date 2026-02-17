import unittest

from the_matrix import Matrix


class MatrixTests(unittest.TestCase):
    def test_matrix_multiplication(self) -> None:
        left = Matrix([[1, 2], [3, 4]])
        right = Matrix([[2, 0], [1, 2]])
        product = left @ right
        self.assertEqual(product.to_list(), [[4.0, 4.0], [10.0, 8.0]])

    def test_power_zero_returns_identity(self) -> None:
        matrix = Matrix([[2, 1], [1, 2]])
        self.assertTrue((matrix**0).almost_equal(Matrix.identity(2)))

    def test_power_two(self) -> None:
        matrix = Matrix([[0.9, 0.1], [0.5, 0.5]])
        expected = Matrix([[0.86, 0.14], [0.70, 0.30]])
        self.assertTrue((matrix**2).almost_equal(expected, tolerance=1e-12))

    def test_multiply_row_vector(self) -> None:
        matrix = Matrix([[0.9, 0.1], [0.5, 0.5]])
        result = matrix.multiply_row_vector([1.0, 0.0])
        self.assertEqual(result, [0.9, 0.1])


if __name__ == "__main__":
    unittest.main()
