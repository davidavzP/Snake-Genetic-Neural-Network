import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import math
import random

###--Single Layer Class
class Perceptron(tf.Module):
    def __init__(self, weights, biases, name=None):
        super().__init__(name=name)
        self.weights = tf.Variable(weights, name="weights")
        self.biases = tf.Variable(biases, name='biases')
    def __call__(self, x):
        return tf.matmul(x, self.weights) + self.biases

###--Multi-Layer Perceptron Class
class MLP(tf.Module):
    def __init__(self, network, layers=[], fitness=0.0, name=None):
        super().__init__(name=name)
        self.lstruc = network
        self.layers = layers
        self.fitness = fitness

    ###--NEW--Calling Method
    @tf.function
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            x = tf.sigmoid(x)
        return x

    ###--NEW--Generalized Layer Creation
    def build_model(self):
        layers = []
        for i in range(len(self.lstruc) - 1):
            (w, b) = self.build_dense_layer(self.lstruc[i], self.lstruc[i + 1])
            layers.append(Perceptron(w, b))
        self.layers = layers
        return self

    def build_dense_layer(self, nodes_in, nodes_out):
        weights = tf.random.normal([nodes_in,nodes_out], 0,1, dtype="float32")
        biases = tf.random.normal([nodes_out], 0,1, dtype="float32")
        return (weights, biases)
    
    def get_layers(self):
        return self.layers

#####################
## 32~HIDDENODES~4 ##
#####################
def get_rand_Agent():
    return MLP([32, 20, 12, 4]).build_model()

def get_rand_Agent_wf(fitness):
    return MLP([32, 20, 12, 4], fitness).build_model()

def normalizeFitness(agents):
    s = 0.0
    fitValues = []
    for agent in agents:
        s += agent.brain.fitness
        assert(agent.brain.fitness > 0.0)
        fitValues.append(agent.brain.fitness)
    assert(s > 0.0)     
    fitValues = list(map(lambda x: (s / x), fitValues))
    s = 0.0
    for i in range(len(fitValues)):
        s += fitValues[i]
    assert(s > 0.0)
    fitValues = list(map(lambda x: (x / s), fitValues))
    return fitValues

def getParent(pop):
    if random.random() > 0.5:
        return tournamentSelection(pop)
    else:
        index = wheelSelection(pop)
        assert(index != None)
        return index

def tournamentSelection(pop):
    i1 = random.randint(0,len(pop) - 1)
    i2 = random.randint(0,len(pop) - 1)
    while i1 == i2:
        i2 = random.randint(0, len(pop) - 1)
    if pop[i1].brain.fitness < pop[i2].brain.fitness:
        return i2
    else:
        return i1

def wheelSelection(pop):
    proportions = normalizeFitness(pop)
    cprop = []
    ctotal = 0.0
    for i in range(len(proportions)):
        ctotal += proportions[i]
        cprop.append(ctotal)
    randsel = random.random()
    for i in range(len(cprop)):
        value = cprop[i]
        if value >= randsel:
            return i
    return None

def crossOver(agents, i1, i2):
    p1 = agents[i1]
    p2 = agents[i2]
    fitness = max(p1.fitness, p2.fitness)
    assert(len(p1.get_layers()) == len(p2.get_layers()))
    lp1 = p1.get_layers()
    lp2 = p2.get_layers()
    num_layers = len(lp1)
    layers = p1.lstruc

    wp1 = np.array([])
    bp1 = np.array([])
    wp2 = np.array([])
    bp2 = np.array([])
    for i in range(num_layers):
        wp1 = np.append(wp1, tf.reshape(lp1[i].weights, [-1]).numpy())
        bp1 = np.append(bp1, tf.reshape(lp1[i].biases, [-1]).numpy())
        wp2 = np.append(wp2, tf.reshape(lp2[i].weights, [-1]).numpy())
        bp2 = np.append(bp2, tf.reshape(lp2[i].biases, [-1]).numpy())

    assert(np.shape(wp1) == np.shape(wp2))
    assert(np.shape(bp1) == np.shape(bp2))
    cw = np.array([])
    cb = np.array([])
    for i in range(len(wp1)):
        if random.random() > 0.5:
            cw = np.append(cw, wp1[i])
        else:
            cw = np.append(cw, wp2[i])
    for i in range(len(bp1)):
        if random.random() > 0.5:
            cb = np.append(cb, bp1[i])
        else:
            cb = np.append(cb, bp2[i])
    networkweights = []
    networkbiases = []
    startw = 0
    startb = 0
    for i in range(num_layers):
        a = layers[i]
        endb = layers[i + 1]
        endw = startw + a * endb 

        w = cw[startw:endw]
        w = tf.reshape(w, [a, endb])
        networkweights.append(w)
        startw = endw

        endb = startb + endb
        b = cb[startb: endb]
        networkbiases.append(b)
        startb = endb

    network = []
    for i in range(num_layers):
        network.append(Perceptron(networkweights[i], networkbiases[i]))

    return MLP(layers, layers=network, fitness=fitness)

def mutate(agent, chance):
    fitness = agent.fitness
    layers = agent.get_layers()
    num_layers = len(layers)
    network = agent.lstruc

    nw = np.array([])
    nb = np.array([])
    for i in range(num_layers):
        nw = np.append(nw, tf.reshape(layers[i].weights, [-1]).numpy())
        nb = np.append(nb, tf.reshape(layers[i].biases, [-1]).numpy())

    for i in range(len(nw)):
        if random.uniform(0,1) < chance:
                print(nw[i])
                nw[i] += random.gauss(0, 1)
                print("A: ", nw[i])
    for i in range(len(nb)):
        if random.uniform(0,1) < chance:
            nb[i] += random.gauss(0,1)
    
    networkweights = []
    networkbiases = []
    startw = 0
    startb = 0
    for i in range(num_layers):
        a = network[i]
        endb = network[i + 1]
        endw = startw + a * endb 

        w = nw[startw:endw]
        w = tf.reshape(w, [a, endb])
        networkweights.append(w)
        startw = endw

        endb = startb + endb
        b = nb[startb: endb]
        networkbiases.append(b)
        startb = endb

    network = []
    for i in range(num_layers):
        network.append(Perceptron(networkweights[i], networkbiases[i]))
    
    return MLP(layers, layers=network, fitness=fitness)


def normalizeFitnessTest(agents):
    s = 0.0
    agent_scores = []
    for i in range(len(agents)):
        s += agents[i].fitness
    assert(s > 0.0)
    for i in range(len(agents)):
        agents[i].fitness = agents[i].fitness/s
        agent_scores.insert(i, agents[i])
    return agent_scores

def chooseAgentTest(agent_scores):
    index = 0
    r = random.random()
    while r > 0.0:
        r = r - agent_scores[index].fitness
        index += 1
    index -= 1
    assert(index > -1)
    return index


def test_gnn():
    agents = []
    agents.append(get_rand_Agent_wf(25))
    agents.append(get_rand_Agent_wf(24))
    agents.append(get_rand_Agent_wf(15))
    agents.append(get_rand_Agent_wf(12))
    agents.append(get_rand_Agent_wf(7))
    print("UNORMALIZED")
    for i in range(len(agents)):
        print("AGENT", i,  " FIT: ", agents[i].fitness)
    print("NORMALIZED")
    agents = normalizeFitnessTest(agents)
    for i in range(len(agents)):
        print("AGENT", i,  " FIT: ", agents[i].fitness)
    pool = []
    for i in range(len(agents)):
        n = int(agents[i].fitness * 10)
        n = max(n , 1)
        for j in range(n):
            pool.append(agents[i])
    print("POOL")
    for i in range(len(pool)):
        print("AGENT", i,  " FIT: ", pool[i].fitness)
    indices = []
    for i in range(len(agents)):
        i1 = chooseAgentTest(pool)
        i2 = chooseAgentTest(pool)
        indices.append((i1, pool[i1].fitness, i2, pool[i2].fitness))
    print("PICKED")
    for i in indices:
        print(i)
