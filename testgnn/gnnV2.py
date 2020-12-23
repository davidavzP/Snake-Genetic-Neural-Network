import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import math
import random

class Perceptron(tf.Module):
    def __init__(self, weights, biases, name=None):
        super().__init__(name=name)
        self.weights = tf.Variable(weights, name="weights")
        self.biases = tf.Variable(biases, name='biases')
    def __call__(self, x):
        return tf.matmul(x, self.weights) + self.biases

class MLP(tf.Module):
    def __init__(self, w1, b1, w2, b2, fitness=0.0, name=None):
        super().__init__(name=name)
        self.layer1 = Perceptron(w1, b1)
        self.layer2 = Perceptron(w2, b2)
        self.fitness = fitness
    def __call__(self, x):
        h1 = self.layer1(x)
        h1 = tf.sigmoid(h1)
        out = self.layer2(h1)
        out = tf.sigmoid(out)
        return out
    def get_layers(self):
        return [self.layer1, self.layer2]


def generate_Agents(n):
    agents = []
    for i in range(1, n+1):
        agents.append(get_rand_Agent(i))
    return agents

def get_rand_Agent(i):
    inputs_hidden_w = tf.random.normal([784,15], 0,1, dtype="float32")
    inputs_hidden_b = tf.random.normal([15], 0,1, dtype="float32")
    hidden_outputs_w = tf.random.normal([15,10], 0,1, dtype="float32")
    hidden_outputs_b = tf.random.normal([10], 0,1, dtype="float32")

    return MLP(inputs_hidden_w, inputs_hidden_b, hidden_outputs_w,hidden_outputs_b, 0.0, "agent"+str(i))

def normalizeFitness(agents):
    s = 0.0
    agent_scores = []
    for i in range(len(agents)):
        s += agents[i].brain.fitness
    assert(s > 0.0)
    for i in range(len(agents)):
        agent_scores.insert(i, agents[i].brain.fitness/s)
    return agent_scores

def chooseAgent(agent_scores):
    index = 0
    r = random.uniform(0,1)
    while r > 0.0:
        r = r - agent_scores[index]
        index += 1
    index -= 1
    return index

##ASSUME AGENTS IS SORTED
def crossOver(agents, i1, i2):
    p1 = agents[i1].brain
    p2 = agents[i2].brain
    assert(len(p1.get_layers()) == len(p2.get_layers()))
    lsp1 = p1.get_layers()
    lsp2 = p2.get_layers()
    len_child = len(lsp1) * 2
    child = []
    for i in range(len(lsp1)):
        lp1 = lsp1[i]
        lp2 = lsp2[i]

        wp1 = lp1.weights
        wp1shape = wp1.shape
        bp1 = lp1.biases

        wp2 = lp2.weights
        wp2shape = wp2.shape
        bp2 = lp2.biases

        assert(wp1shape == wp2shape)
        assert(len(wp1shape) == 2)
        lengthw = wp1shape[0] * wp1shape[1]
        assert(bp1.shape == bp2.shape)
        bshape = bp1.shape
        assert(len(bp1.shape) == 1)
        lengthb = bshape[0]

        wt1 = tf.reshape(wp1, [-1])
        wt2 = tf.reshape(wp2, [-1])

        splitw = random.randint(0, lengthw)
        splitb = random.randint(0, lengthb)

        wt1 = wt1[splitw:]
        wt2 = wt2[:splitw]

        bp1 = bp1[splitb:]
        bp2 = bp2[:splitb]

        w = tf.concat([wt1, wt2], 0)
        w = tf.reshape(w, wp1shape)
        b = tf.concat([bp1, bp2], 0)
        assert(w.shape == wp1shape)
        assert(b.shape == bshape)
        child.append(w)
        child.append(b)
    assert(len(child) == len_child)
    child = MLP(child[0], child[1], child[2], child[3])
    return child 

def mutate(agent, chance):
    fitness = agent.fitness
    layers = agent.get_layers()
    len_child = len(layers)*2
    child = []
    for i in range(len(layers)):
        weights = layers[i].weights
        wshape = weights.shape
        biases = layers[i].biases
        bshape = biases.shape
        tweights = tf.reshape(weights, [-1]).numpy()
        tbiases = tf.reshape(biases, [-1]).numpy()
        wob = random.randint(0,1)
        if wob == 1:
            for i in range(len(tweights)):
                if random.uniform(0,1) < chance:
                    tweights[i] = random.uniform(-tweights[i] - 3, tweights[i] + 3)
        else:
            for i in range(len(tbiases)):
                if random.uniform(0,1) < chance:
                    tbiases[i] = random.uniform(-tbiases[i] - 3, tbiases[i] + 3)
        w = tf.convert_to_tensor(tweights, dtype="float32")
        b = tf.convert_to_tensor(tbiases, dtype="float32")
        w = tf.reshape(w, wshape)
        b = tf.reshape(b, bshape)
        child.append(w)
        child.append(b)
    assert(len_child == 4)
    child = MLP(child[0], child[1], child[2], child[3], fitness)
    return child



# (train, lbtrain), (test, lbtest) = keras.datasets.mnist.load_data()
# train = train.reshape(60000, 784).astype("float32") / 255
# test = test.reshape(10000, 784).astype("float32") / 255
# lbtrain = tf.keras.utils.to_categorical(lbtrain, num_classes=10, dtype="float32")
# lbtest = tf.keras.utils.to_categorical(lbtest, num_classes=10, dtype="float32")

# def test_gnn(x, lbx):
#     population = []
#     n = 100
#     population.extend(generate_Agents(n))
#     generations = 100
#     mutation_rate = 0.02
#     for i in range(generations):
#         population = feedForwardAgents(population, x, lbx)
#         population = sorted(population, key=lambda agent: agent.fitness, reverse=True)
#         print("GENERATION ", i, "'s BEST FITNESS: ", population[0].fitness)
#         pool = []
#         for i in range(len(population)):
#             n = int(population[i].fitness * 100)
#             for j in range(n):
#                 pool.append(population[i])
#         fitprop = normalizeFitness(pool)
#         for i in range(len(population)):
#             i1 = chooseAgent(fitprop)
#             i2 = chooseAgent(fitprop)
#             child = crossOver(pool,i1,i2)
#             child = mutate(child, mutation_rate)
#             population[i] = child

#     population = feedForwardAgents(population, x, lbx)
#     population = sorted(population, key=lambda agent: agent.fitness, reverse=True)
#     return population[0]

# best = test_gnn(train, lbtrain)
# print("BEST FIT: ", best.fitness)

# def evalute_output(model, x, y):
#     correct = 0.0
#     for i in range(len(x)):
#         inputs = x[i]
#         guess = model([inputs])
#         actual = [y[i]]
#         a = tf.keras.metrics.categorical_accuracy(actual, guess)
#         if a == 1.0:
#             correct += 1.0
#     total = float(len(x))
#     accuracy = (correct / total) * 100.0
#     print("C: ", correct, "ACCURACY: ", accuracy)

# print("AFTER TRAIN EVAL: ")
# evalute_output(best, test, lbtest)
  
        
        
