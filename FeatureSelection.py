"""
service for feature selection
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr


def select_features_by_mutual_info(x, y, n_features_to_select=5):
    # Compute mutual information between each feature and the target variable
    mi_scores = mutual_info_regression(x, y)

    # Select features with highest mutual information scores
    feature_indices = np.argsort(mi_scores)[::-1][:n_features_to_select]

    # Get feature names from original DataFrame
    feature_names = x.columns

    # Return selected feature names and data
    return feature_names[feature_indices], x.iloc[:, feature_indices]


def select_features_by_pcc(x, y, pcc):
    """
    :param x: pd.DataFrame
    :param y:
    :param pcc: float
    :return:
    """
    # remove no variance features
    variance = x.var(axis=0)
    zero_variance_cols = variance[variance == 0].index.tolist()
    x = x.drop(zero_variance_cols, axis=1)
    # calculate abs value of corr between features
    corr = abs(x.corr())
    features_names = list(x.columns)
    feature_num = x.shape[1]
    # set self.pcc == 0
    for i in range(feature_num):
        corr[features_names[i]][features_names[i]] = 0

    while True:
        # get max pcc
        max_corr = corr.max().max()

        if max_corr <= pcc or x.shape[1] < 2:  # stop when only one feature
            break

        # find the features should be deleted
        col1, row1 = np.where(corr == max_corr)

        pcc1 = abs(pearsonr(x[features_names[row1[0]]], y)[0])
        pcc2 = abs(pearsonr(x[features_names[col1[0]]], y)[0])
        if pcc1 >= pcc2:
            # print(pcc1, pcc2)
            drop_feature = corr.index[col1].tolist()[0]
        else:
            # print(pcc1, pcc2)
            drop_feature = corr.index[row1].tolist()[0]

        print(drop_feature)
        x = x.drop(drop_feature, axis=1)
        corr = abs(x.corr())
        feature_num = x.shape[1]
        features_names = list(x.columns)
        for i in range(feature_num):
            corr[features_names[i]][features_names[i]] = 0

    selected_features = list(x.columns)
    return selected_features

    # wrapper


def select_features_by_ga(x, y, task="cls"):
    """
    feature selection by genetic algorithm
    :param x:
    :param y:
    :return:
    """
    import random
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.model_selection import train_test_split

    def fitness_function(features, X_train, y_train, X_test, y_test, task=task):
        # Fit a logistic regression model using the selected features
        if task == 'cls':
            model = LogisticRegression()
            model.fit(X_train[:, features], y_train)
            # Calculate the accuracy on the test set
            score = model.score(X_test[:, features], y_test) - 0.001 * len(features)

            # Return the fitness score
            return score
        else:
            model = LinearRegression()
            model.fit(X_train[:, features], y_train)
            score = model.score(X_test[:, features], y_test) - 0.001 * len(features)
            return score

    def initialize_population(num_features, population_size):
        # Generate an initial population randomly
        population = []
        for _ in range(population_size):
            chromosome = [random.randint(0, 1) for _ in range(num_features)]
            population.append(chromosome)
        return population

    def mutation(chromosome, mutation_rate):
        # Perform mutation by randomly flipping bits in the chromosome
        for i in range(len(chromosome)):
            if random.random() < mutation_rate:
                chromosome[i] = 1 - chromosome[i]
        return chromosome

    def crossover(parent1, parent2):
        # Perform crossover by selecting a random crossover point
        # and swapping the genetic material between parents
        crossover_point = random.randint(0, len(parent1))
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def select_parents(population, fitness_scores):
        # Select two parents using tournament selection
        tournament_size = 5
        tournament_indices = random.sample(range(len(population)), tournament_size)
        parent1_idx = min(tournament_indices, key=lambda x: fitness_scores[x])
        tournament_indices.remove(parent1_idx)
        parent2_idx = min(tournament_indices, key=lambda x: fitness_scores[x])
        return population[parent1_idx], population[parent2_idx]

    def genetic_feature_selection(X, y, population_size=30, num_generations=20, mutation_rate=0.05, task=task):
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Initialize the population
        num_features = X.shape[1]
        population = initialize_population(num_features, population_size)

        for generation in range(num_generations):
            # Evaluate the fitness of each chromosome
            fitness_scores = [fitness_function(features, X_train, y_train, X_test, y_test, task=task) for features
                              in
                              population]

            # Select parents for crossover
            parents = [select_parents(population, fitness_scores) for _ in range(population_size // 2)]

            # Create offspring through crossover and mutation
            offspring = []
            for parent1, parent2 in parents:
                child1, child2 = crossover(parent1, parent2)
                child1 = mutation(child1, mutation_rate)
                child2 = mutation(child2, mutation_rate)
                offspring.extend([child1, child2])

            # Replace the old population with the new offspring
            population = offspring

        # Select the best chromosome as the selected features
        best_chromosome = max(population,
                              key=lambda x: fitness_function(x, X_train, y_train, X_test, y_test, task=task))
        print(best_chromosome)
        best_score = fitness_function(best_chromosome, X_train, y_train, X_test, y_test, task=task)
        return best_chromosome, best_score

    best_chromosome, best_score = genetic_feature_selection(x, y)
    return best_chromosome, best_score
