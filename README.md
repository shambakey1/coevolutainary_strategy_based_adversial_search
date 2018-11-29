# adversial_project

This is an implementation of the Adversial Search project of Udacity AI Nanodegree. The implementation uses co-evolutionary 
strategy optimization algorithm to find the sub-optimal move of current player.

The co-evolutionary strategy is similar to 
Monte Carlo Tree Search (MCTS) in that it runs many simulations of the game to decide the next move of current player. Also, 
the co-evolutionary strategy is similar to Genetic Algorithm and Evolutionary Strategy optimization techniques in that it uses 
crossover, mutation and selection.

PS: The implmenetation uses files from the DEAP python module (https://github.com/DEAP/deap) with some modifications. I modified the "eaSimple" algorithm in the DEAP library to "eaSimple_mod" to suit the project/s knights isolation problem. Unfortunatelly, the implementation consumes too much memory than what is required by the project (maybe due to the numpy library)
