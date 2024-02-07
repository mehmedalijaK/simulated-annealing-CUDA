# Simulated Annealing Implementation for Image Optimization in the CUDA/PyCUDA environment.

## Introduction
This project is developed for the Parallel Algorithms course at the faculty.
Simulated Annealing is an optimization algorithm used to find approximate optimal solutions for large and complex problems. This algorithm is inspired by the annealing process in metallurgy, where metal is heated and then slowly cooled to improve its structure and reduce internal energy.

## Key Components of Simulated Annealing Algorithm
1. **Initial Solution:** The algorithm begins with a random solution to the problem.
2. **Testing Neighbor Solutions:** At each step, the algorithm generates a new solution that is "close" to the current one. This is often a random modification of the current solution.
3. **Acceptance Criterion:** Based on the difference in "quality" between the current and generated solutions, as well as the current "temperature," the algorithm decides whether to accept the new solution:
   - Better solutions are always accepted.
   - Even worse solutions may be accepted with a certain probability, which helps avoid local minima.
4. **Temperature:** Represents a parameter that controls the probability of accepting solutions that are worse than the current one. The high temperature allows the algorithm to explore a wider range of solutions, including the worse one.
5. **Cooling:** Gradually reducing the temperature leads the algorithm to become less likely to accept poorer solutions. This is analogous to the cooling process in metallurgy.
6. **Iteration:** The process is repeated until the temperature reaches a predefined minimum value or until other stopping criteria are met (such as the number of iterations).

## Application of the Algorithm
Simulated Annealing is used in various fields, including task scheduling, network design, route optimization, financial modeling, and many other complex optimization problems. The main advantage of this algorithm is its ability to efficiently explore large solution spaces and find good solutions for problems that are too complex for exact solutions.

## Simulated Annealing and Parallelism
The basic algorithm is sequential, but there are several ways to introduce parallelism:
- Parallelization of solution quality evaluation – typically involves computing some value that can be parallelized.
- Modification of the algorithm where multiple neighboring solutions are tested in parallel in each iteration, and the best one is chosen.
- Parallel execution of multiple optimization processes (perhaps with different initial states, random number generators, temperature settings, etc.). Optionally, we can occasionally check which process is performing the best and discard the poorer ones.

## Levels of Parallelism
1. Threads along the x-dimension of the block should be used to compute the change in image energy during swapping.
2. Each of the 12 threads in the block calculates the energy of one pixel before and after the swap. Write the result to shared memory.
3. The total change in energy may be computed using one thread.
4. Threads along the y-dimension of the block should be used to calculate multiple alternative moves, from which the best one will be chosen.
5. Keep the matrix with pixel values in shared memory.
6. Store the matrix as uint8 (unsigned char), but be careful of overflow when calculating differences between neighboring pixels.
7. Threads of different blocks execute parallel, independent optimization processes.



