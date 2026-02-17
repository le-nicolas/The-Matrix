# The-Matrix

A small from-scratch Python project for learning matrix operations through a practical Markov chain simulator.

## What is included

- `the_matrix/matrix.py`: lightweight matrix class with multiplication, transpose, identity, and fast exponentiation.
- `the_matrix/markov.py`: Markov chain model with transition validation, simulation, n-step transitions, and stationary distribution estimation.
- `Markov.py`: runnable weather simulation example.
- `tests/`: unit tests for matrix math and Markov behavior.

## Quick Start

```bash
python Markov.py
```

Example output:

```text
Weather Markov Chain Simulation
Step  0: [1.000, 0.000, 0.000]  likely=Sunny
Step  1: [0.700, 0.200, 0.100]  likely=Sunny
...
```

## Run Tests

```bash
python -m unittest discover -s tests -v
```

## Why matrices for Markov chains?

If `x_t` is the row-vector distribution at step `t`, and `P` is the transition matrix:

`x_(t+1) = x_t P`

For multi-step transitions:

`x_(t+n) = x_t P^n`

This project keeps that relationship explicit in code.

