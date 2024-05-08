import string as str
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

# Fitness Function
def calcFitness(target, value):
    return np.mean(np.array(list(target)) == np.array(list(value)))

# generate Chromosome
def genChromosome(size, values):
    return ''.join(rnd.choice(values) for i in range(size));

# Population
def generatePopulation(amount, length, values):
    pop = [];
    for i in range(amount):
        pop.append(genChromosome(length, values));
    return pop;

# Parent selection function
def tournamentSelection(size, population, fitness):
    participant = rnd.sample(range(0, len(population)),size);
    winner = 0;
    score = 0;
    for i in range(len(participant)):
        if fitness[participant[i]] > score:
            winner = participant[i];
            score = fitness[participant[i]];
    return population[winner];

# Single Point
def singlePointCrossover(parent1, parent2):
    crosspoint = rnd.randint(1, len(parent1)-1);
    c1 = parent1[:crosspoint] + (parent2[crosspoint:]);
    c2 = parent2[:crosspoint] + (parent1[crosspoint:]);
    return (c1, c2);

# Mutation function
def mutate(child, mutation_probability, genes):
    chernobyl_child = "";

    for char in child:
        if rnd.uniform(0, 1) > mutation_probability:
            chernobyl_child += char;
        else:
            chernobyl_child += rnd.choice(genes);
    
    return chernobyl_child;

# Generate children
def makeChildren(parent1, parent2, crossover_probability, mutation_probability, genes):
    # Crossover
    if rnd.uniform(0, 1) <= crossover_probability:
        child1, child2 = singlePointCrossover(parent1, parent2);
    else:
        child1 = parent1;
        child2 = parent2;

    # Mutate
    child1 = mutate(child1, mutation_probability, genes);
    child2 = mutate(child2, mutation_probability, genes);
    
    return (child1, child2);


def RunGA(target, population_size, tournament_pool, genes, crossover_probability, mutation_probability, max_generations):

    # Generate population
    population = generatePopulation(population_size, len(target), genes)
    best_canditates = [];
    best_scores = [];
    generations = [];

    message = "│ Population: {} │ Max Generations: {} │ Crossover probability: {} │ Mutation probability: {} │ Tournament participants: {} │".format(population_size, max_generations, crossover_probability, mutation_probability, tournament_pool);
    print("─"*len(message))
    print(message)
    print("─"*len(message))


    generation = 1;
    while(True):
        
        # Calculate fitness of population
        fitness = [calcFitness(target, chromosome) for chromosome in population];
        
        # Get score
        best_score = max(fitness);
        best_canditate = population[fitness.index(best_score)];

        best_canditates.append(best_canditate);
        best_scores.append(best_score);
        generations.append(generation);

        print("GEN: {:02d} || Best Canditate: {} | score: {}".format(generation, best_canditate, best_score));

        # stopping condition
        if generation > max_generations-1 or best_score == 1:
            break;

        # new empty population
        new_population = [];

        # Loop until new population is the correct size
        while len(new_population) < len(population):

            # Select parents
            parent1 = tournamentSelection(tournament_pool, population, fitness);
            parent2 = tournamentSelection(tournament_pool, population, fitness);

            # Create children
            child1, child2 = makeChildren(parent1, parent2, crossover_probability, mutation_probability, genes);

            # add to new population
            new_population.append(child1);
            new_population.append(child2);
        
        # Set the new population to the actual population
        population = new_population;
        generation+=1;
    return (best_scores, best_canditates, generations)

genes = "ABCDEFGHIJKLMNOPQRSTUVWXYZÆØÅabcdefghijklmnopqrstuvwxyzæøå0123456789*_-'";
best_scores, best_candidates, generations = RunGA("Vegard_Forde*581814", 500, 50, genes, 0.5, 0.02, 100);

plt.plot(generations, best_scores);
plt.show();