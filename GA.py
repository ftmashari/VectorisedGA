import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


def fitness(route):
    route = list(route)
    route.insert(0, 0)
    route.append(0)
    xy = cities.iloc[route]
    sub = xy - xy.shift(1)
    subsquared = sub[:-1] ** 2
    euclid = np.sqrt(subsquared.sum(1))
    dist = sum(euclid)
    return dist


def createinitialpopulation(pop, initialroute):
    length = len(initialroute)
    initpop = pd.DataFrame(pd.np.empty((pop, length)) * pd.np.nan)
    initpop = initpop.apply(lambda _: random.sample(initialroute, length), axis=1)
    initpop.reset_index(inplace=True)
    initpop.drop('index', axis=1, inplace=True)
    return initpop


def selectelites(population, num):
    distances = population.apply(fitness, axis=1)
    distances.sort_values(inplace=True)
    population = population.reindex(distances.index)
    population = population.head(int(num))
    population.reset_index(inplace=True)
    population.drop('index', axis=1, inplace=True)
    return distances.head(1), population


def combine(num, *dataframes):
    result = pd.concat(dataframes)
    result.reset_index(inplace=True)
    result.drop('index', axis=1, inplace=True)
    bestdistance, result = selectelites(result, num)
    return bestdistance, result


def mutate(individual):
    num = individual.size
    loc1, loc2 = random.sample(range(num), 2)
    individual[loc1], individual[loc2] = individual[loc2], individual[loc1]
    return individual


def swap(x, y):
    length = x.size
    loc = random.randint(0, length - 1)
    temp = x[~x.isin(y[0:loc])].dropna()
    x[0:loc] = y[0:loc]
    x[loc:] = temp.values
    return x


def crossover(population):
    nrow, ncol = population.shape
    firstrow = population.iloc[0, :].copy()
    for i in range(int(nrow-1)):
        population.iloc[i, :] = swap(population.iloc[i, :], population.iloc[i+1, :])
    i = nrow - 1
    population.iloc[i,:] = swap(population.iloc[i, :], firstrow)
    return population


def genetic(pop, gen, mutationrate, crossoverrate):
    initialroute = list(cities.index.values[1:])
    population = createinitialpopulation(pop, initialroute)
    nummutation = int(np.ceil(mutationrate * pop))
    numcrossover = int(np.ceil(crossoverrate * pop))
    numelite = int(np.ceil((1-(mutationrate+crossoverrate)) * pop))
    _, elites = selectelites(population, numelite)
    bestdistances = []
    for i in range(0, gen):
        print('generation ' + str(i+1))
        population_mut = population.sample(n=nummutation)
        mutated = population_mut.apply(mutate, axis=1)
        population_cro = population.sample(n=numcrossover)
        breed = crossover(population_cro)
        bestdistance, population = combine(pop, elites, mutated, breed)
        elites = population.head(numelite)
        bestdistances.append(bestdistance)
    _, best = selectelites(population, 1)
    return bestdistances, best


def visualise(distances):
    f = plt.figure()
    plt.plot(distances.index+1, distances, '-o')
    plt.xlabel('Generation')
    plt.ylabel('Distance')
    f.savefig('performance.pdf', bbox_inches='tight')


def main():
    bestdistances, best = genetic(1000, 100, 0.05, 0.8)
    best = best.transpose().append([0])
    outpath = 'path.csv'
    best.to_csv(outpath, index=False, mode='a')
    visualise(pd.Series(bestdistances))


cities = pd.read_csv('cities.csv', index_col=0)
if __name__ == '__main__':
    main()