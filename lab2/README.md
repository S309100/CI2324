Lab 2: Nim Game Agents

Task 2.1: An agent using fixed rules based on nim-sum

For Task 2.1, we implemented a deterministic agent that uses the nim-sum to make strategic moves in the game of Nim. The strategy, defined in the rule_based_nim_sum function, calculates the current nim-sum and attempts to make a move that results in a zero nim-sum, which is the optimal play. If the nim-sum is already zero, indicating a disadvantageous position, the agent defaults to a random move.

Task 2.2: An agent using evolved rules using ES

Task 2.2 introduces an evolutionary strategy (ES) agent, which evolves its parameters over successive games to improve performance. The evolve_strategy function manages the evolution process, selecting the most fit individuals and applying mutations to explore the strategy space. The evolved strategy is encapsulated in the evolved_strategy function, which uses the best parameters obtained from the evolution to make game decisions.

Instructions

To run the game agents, execute nim.py. Make sure that the required libraries, particularly numpy, are installed in your environment. For observing the evolutionary progress, ensure that the logging level is set to INFO.

File Structure

nim.py: This script contains the implementation of the nim-sum-based and random agents for Task 2.1, along with the code required to play the game of Nim by rule_based_nim_sum, and evolved_strategy
Contains the evolutionary strategy logic for Task 2.2, which evolves a population of strategies to find the most effective one against a rule-based nim-sum agent.

Collaborations

This project was developed by Samaneh Gharehdagh Sani and Hossein Kakavand.

References

The evolutionary strategy was inspired by genetic algorithms and principles from evolutionary computation. No direct external sources or libraries were used beyond what is provided by Python's standard library and numpy for numerical operations.

