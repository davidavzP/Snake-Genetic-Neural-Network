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
    def __init__(self, w1, b1, w2, b2, w3, b3, fitness=0.0, name=None):
        super().__init__(name=name)
        self.layer1 = Perceptron(w1, b1)
        self.layer2 = Perceptron(w2, b2)
        self.layer3 = Perceptron(w3, b3)
        self.fitness = fitness
    def __call__(self, x):
        x = tf.convert_to_tensor([x], dtype="float32")
        h1 = self.layer1(x)
        h1 = tf.nn.relu(h1)
        h2 = self.layer2(h1)
        h2= tf.nn.relu(h2)
        out = self.layer3(h2)
        out = tf.sigmoid(out)
        return out
    def get_layers(self):
        return [self.layer1, self.layer2,self.layer3]

#l1 = 640w, 20b
#l2 = 240w, 12b
#l3 = 48w, 4b

#0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610

def get_rand_Agent():
    inputs_hidden_w = tf.random.uniform([32,20], -1,1, dtype="float32")
    inputs_hidden_b = tf.random.uniform([20], -1,1, dtype="float32")
    hidden1_hidden2_w = tf.random.uniform([20,12], -1,1, dtype="float32")
    hidden1_hidden2_b = tf.random.uniform([12], -1,1, dtype="float32")
    hidden2_outputs_w = tf.random.uniform([12,4], -1,1, dtype="float32")
    hidden2_outputs_b = tf.random.uniform([4],-1,1, dtype="float32")


    return MLP(inputs_hidden_w, inputs_hidden_b, hidden1_hidden2_w, hidden1_hidden2_b, hidden2_outputs_w, hidden2_outputs_b)

def get_rand_Agent_wf(fitness):
    inputs_hidden_w = tf.random.uniform([32,24], -1,1, dtype="float32")
    inputs_hidden_b = tf.random.uniform([24], -1,1, dtype="float32")
    hidden1_hidden2_w = tf.random.uniform([24,10], -1,1, dtype="float32")
    hidden1_hidden2_b = tf.random.uniform([10], -1,1, dtype="float32")
    hidden2_outputs_w = tf.random.uniform([10,4], -1,1, dtype="float32")
    hidden2_outputs_b = tf.random.uniform([4], -1,1, dtype="float32")


    return MLP(inputs_hidden_w, inputs_hidden_b, hidden1_hidden2_w, hidden1_hidden2_b, hidden2_outputs_w, hidden2_outputs_b, fitness)

def normalizeFitness(agents):
    s = 0.0
    agent_scores = []
    for i in range(len(agents)):
        s += agents[i].brain.fitness
    assert(s > 0.0)
    for i in range(len(agents)):
        agents[i].brain.fitness = agents[i].brain.fitness/s
        agent_scores.insert(i, agents[i])
    return agent_scores

def chooseAgent(agent_scores):
    index = 0
    r = random.uniform(0,1)
    while r > 0.0:
        r = r - agent_scores[index].brain.fitness
        index += 1
    index -= 1
    assert(index > -1)
    return index

##ASSUME AGENTS IS SORTED
def crossOver(agents, i1, i2):
    p1 = agents[i1].brain
    p2 = agents[i2].brain
    fitness = max(p1.fitness, p2.fitness)
    assert(len(p1.get_layers()) == len(p2.get_layers()))
    lsp1 = p1.get_layers()
    lsp2 = p2.get_layers()
    len_child = len(lsp1) * 2
    child = []
    ksplit = 13

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
        print(wt1)
        wt2 = tf.reshape(wp2, [-1])

        splitw = random.randint(1, lengthw-1)
        splitb = random.randint(1, lengthb-1)

        wt1 = wt1[splitw:]
        wt2 = wt2[:splitw]

        bp1 = bp1[splitb:]
        bp2 = bp2[:splitb]
    
        w = tf.concat([wt2, wt1], 0)
        w = tf.reshape(w, wp1shape)
        b = tf.concat([bp2, bp1], 0)
        assert(w.shape == wp1shape)
        assert(b.shape == bshape)
        child.append(w)
        child.append(b)

    assert(len(child) == len_child)

    child = MLP(child[0], child[1], child[2], child[3], child[4], child[5], fitness)
    return child 

def crossOver2(agents, i1, i2):
    p1 = agents[i1].brain
    p2 = agents[i2].brain
    fitness = max(p1.fitness, p2.fitness)
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

        wt1 = tf.reshape(wp1, [-1]).numpy()
        wt2 = tf.reshape(wp2, [-1]).numpy()

        permw = np.random.permutation(lengthw)

        i = 0
        j = permw[i]
        c1 = [i]


        while(j != c1[0]):
            i = j
            j = permw[i]
            #a1.remove()
            c1.append(i)

        c1 = sorted(c1)
                
        w = []

        i = 0
        j = 0
        while(i < len(c1)):
            v = c1[i]
            x = wt1[v]
            v2 = v + 1
            i = i + 1
            w.extend([x])
            if i < len(c1):
                v3 = c1[i]
                y = wt2[v2:v3]
                w.extend(y)
            else:
                if v < lengthw - 1:
                    w.extend(wt2[v+1:])

        #splitw = random.randint(1, lengthw-1)
        

        #wt1 = wt1[splitw:]
        #wt2 = wt2[:splitw]

        splitb = random.randint(1, lengthb-1)

        bp1 = bp1[splitb:]
        bp2 = bp2[:splitb]
    
        #w = tf.concat([wt2, wt1], 0)
        #w = tf.reshape(w, wp1shape)
        w = tf.convert_to_tensor([w], dtype="float32")
        w = tf.reshape(w, wp1shape)
        b = tf.concat([bp2, bp1], 0)
        assert(w.shape == wp1shape)
        assert(b.shape == bshape)
        child.append(w)
        child.append(b)

    assert(len(child) == len_child)

    child = MLP(child[0], child[1], child[2], child[3], child[4], child[5], fitness)
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
        
        for i in range(len(tweights)):
            if random.uniform(0,1) < chance:
                tweights[i] += random.gauss(0, 1)
        for i in range(len(tbiases)):
            if random.uniform(0,1) < chance:
                tbiases[i] = random.gauss(0, 1)

        w = tf.convert_to_tensor(tweights, dtype="float32")
        b = tf.convert_to_tensor(tbiases, dtype="float32")
        w = tf.reshape(w, wshape)
        b = tf.reshape(b, bshape)
        child.append(w)
        child.append(b)
    assert(len(child) == len_child)
    child = MLP(child[0], child[1], child[2], child[3], child[4], child[5], fitness)
    return child

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



