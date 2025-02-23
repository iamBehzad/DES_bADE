# DES-bADE
A Metaheuristic-Driven Dynamic Ensemble Selection Method

# Abstract
Dynamic Ensemble Selection (DES) techniques aim to improve classification accuracy by selecting the most competent classifiers for each test instance. However, traditional DES methods typically define the Region of Competence (RoC) using heuristic-based approaches such as K-Nearest Neighbors (KNN) or clustering, which may not always yield optimal results. In this paper, we propose DES-bADE2, a novel Metaheuristic-Driven Dynamic Ensemble Selection method that utilizes a binary Advanced Differential Evolution (bADE) algorithm to optimize the selection of the RoC dynamically. Instead of relying on a fixed selection mechanism, DES-bADE evolves an optimal subset of neighbors based on multiple competence criteria, including accuracy, diversity, confidence, output profiles, and distance metrics. By leveraging binary ADE with chaotic initialization and evolutionary search, DES-bADE adaptively selects the most informative and complementary classifiers for each test instance. Experimental evaluations on 30 benchmark datasets demonstrate that DES-bADE outperforms state-of-the-art DES methods in terms of classification accuracy. Statistical validation using the Wilcoxon signed-rank and Fridman test confirms the significance of these improvements. The results highlight the potential of metaheuristic-driven optimization in dynamic ensemble selection, making DES-bADE a promising approach for high-accuracy classification in complex and evolving data environments.

