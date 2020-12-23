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
    def __init__(self, w1, b1, w2, b2, name=None):
        super().__init__(name=name)
        self.layer1 = Perceptron(w1, b1)
        self.layer2 = Perceptron(w2, b2)
        self.fitness = 0.0
    def __call__(self, x):
        h1 = self.layer1(x)
        h1 = tf.sigmoid(h1)
        out = self.layer2(h1)
        out = tf.sigmoid(out)
        return out
    def get_layers(self):
        return [self.layer1, self.layer2]

def loss(target_y, predicted_y):
  return tf.reduce_mean(tf.square(target_y - predicted_y))

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

    return MLP(inputs_hidden_w, inputs_hidden_b, hidden_outputs_w,hidden_outputs_b, "agent"+str(i))

def feedForwardAgents(agents, x, lbx):
    for agent in agents:
        lss = loss(lbx, agent(x))
        agent.fitness = lss
    return agents

def crossOver(agents):
    agents = sorted(agents, key=lambda agent: agent.fitness, reverse=False)
    i1 = random.randint(1, len(agents) - 1)
    parent1 = agents[0]
    parent2 = agents[i1]

    p1_layers = parent1.get_layers()
    p2_layers = parent2.get_layers()
    assert(len(p1_layers) == len(p2_layers))
    child1 = []
    child2 = []

    for i in range(len(p1_layers)):
        p1w = p1_layers[i].weights
        p1b = p1_layers[i].biases
        p2w = p2_layers[i].weights
        p2b = p2_layers[i].biases
        assert(p1w.shape == p2w.shape)
        assert(p1b.shape == p2b.shape)
        wshape = p1w.shape
        bshape = p1b.shape
        assert(len(p1w.shape) == len(p1_layers))
        assert(len(bshape) == len(p1_layers) - 1)
        nxt_node_num = wshape[1]
        b_nodes = bshape[0]
        assert(nxt_node_num >= 1)
        wsplit = random.randint(1, nxt_node_num - 1)
        bsplit = random.randint(1, b_nodes - 1)
        p1s1 = p1w[:wshape[0], :wsplit]
        p1s2 = p1w[:wshape[0], wsplit:]
        p1b1 = p1b[:bsplit]
        p1b2 = p2b[bsplit:]

        p2s1 = p2w[:wshape[0], :wsplit]
        p2s2 = p2w[:wshape[0], wsplit:]
        p2b1 = p2b[:bsplit]
        p2b2 = p2b[bsplit:]

        c1 = tf.concat([p1s1, p2s2], 1)
        c2 = tf.concat([p2s1, p1s2], 1)
        c1b = tf.concat([p1b1, p2b2], 0)
        c2b = tf.concat([p2b1, p1b2], 0)
        assert(c1.shape == c2.shape)
        assert(c1.shape == p1w.shape)
        assert(c1b.shape == c2b.shape)
        assert(c1b.shape == p1b.shape)
        child1.append(c1)
        child1.append(c1b)
        child2.append(c2)
        child2.append(c2b)

    assert(len(child1) == len(child2))
    assert(len(child1) == 4)
    child1 = MLP(child1[0], child1[1], child1[2], child1[3], parent1.name + "c")
    child2 = MLP(child2[0], child2[1], child2[2], child2[3], parent1.name + "c")
    agents.pop()
    agents.pop()
    agents.append(child1)
    agents.append(child2)
    return agents

def mutation(agents):
    newagents = []
    for agent in agents:
        a1 = agent
        if random.uniform(0.0, 1.0) <= 0.3:
            a1 = mutate(agent)
        newagents.append(a1)
    return newagents

def mutate(agent):
    layers = agent.get_layers()
    values = []
    for layer in layers:
        w = layer.weights
        b = layer.biases
        wshape = w.shape
        bshape = b.shape
        w = tf.reshape(w, [-1])
        b = tf.reshape(b, [-1])
        w = w.numpy()
        b = b.numpy()
        randw = random.randint(0, len(w) - 1)
        randb = random.randint(0, len(b) - 1)
        wv = random.uniform(-1.5, 1.5)
        bv = random.uniform(-1.5, 1.5)
        w[randw] = wv
        b[randb] = bv
        w = tf.convert_to_tensor(w, dtype="float32")
        b = tf.convert_to_tensor(b, dtype="float32")
        w = tf.reshape(w, wshape)
        b = tf.reshape(b, bshape)
        assert(w.shape ==  wshape)
        assert(b.shape == bshape)
        values.append(w)
        values.append(b)
    assert(len(values) == 4)
    return MLP(values[0], values[1], values[2], values[3])


(train, lbtrain), (test, lbtest) = keras.datasets.mnist.load_data()
train = train.reshape(60000, 784).astype("float32") / 255
test = test.reshape(10000, 784).astype("float32") / 255
lbtrain = tf.keras.utils.to_categorical(lbtrain, num_classes=10, dtype="float32")
lbtest = tf.keras.utils.to_categorical(lbtest, num_classes=10, dtype="float32")


def test_gnn(agents, x, lbx, generations=100):
    population = agents
    population = feedForwardAgents(population, x, lbx)
    for i in range(generations):
        population = crossOver(population)
        population = mutation(population)
        population = feedForwardAgents(population, x, lbx)
        population = sorted(population, key=lambda agent: agent.fitness, reverse=False)
        if population[0].fitness < 0.1:
            print("Got Good at Generation: ", i)
        print("GENRATION ", i, " COMPLETE", " FITTEST: ", population[0].fitness)
    population = sorted(population, key=lambda agent: agent.fitness, reverse=False)
    return population[0]

best = test_gnn(generate_Agents(100), train, lbtrain, 5000)
print("Final Fit: ", best.fitness)


batch_size = 10

train_dataset = tf.data.Dataset.from_tensor_slices((train, lbtrain))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)


def train(model, x, y, epoch, learning_rate):

    with tf.GradientTape() as t:
        # Trainable variables are automatically tracked by GradientTape
        current_loss = loss(y, model(x))

        # Use GradientTape to calculate the gradients with respect to w1,b1,w2,b2
        dw1, db1, dw2, db2 = t.gradient(current_loss, [model.layer1.weights, model.layer1.biases, model.layer2.weights, model.layer2.biases])

        # Subtract the gradient scaled by the learning rate
        model.layer1.weights.assign_sub(learning_rate * dw1)
        model.layer1.biases.assign_sub(learning_rate * db1)

        model.layer2.weights.assign_sub(learning_rate * dw2)
        model.layer2.biases.assign_sub(learning_rate * db2)


    # current_loss = loss(y, model(x))
    # losses.append(current_loss) 
    # print("Epoch %2d: loss=%2.5f" %
    #         (epoch, current_loss))

##EPOCH LOOPS##

losses = []


def training_loop(model, train_dataset, learning_rate=0.9):
    epochs = 10
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            train(model, x_batch_train, y_batch_train, epoch, learning_rate)


# # Define a training loop
# def training_loop(model, x, y):
#     for epoch in epochs:
#         train(model, x, y, learning_rate=3.0)

#         current_loss = loss(y, model(x))
#         losses.append(current_loss) 

#         print("Epoch %2d: loss=%2.5f" %
#             (epoch, current_loss))
print("TRAIN")
training_loop(best, train_dataset)   

def evalute_output(model, x, y):
    correct = 0.0
    for i in range(len(x)):
        inputs = x[i]
        guess = model([inputs])
        actual = [y[i]]
        a = tf.keras.metrics.categorical_accuracy(actual, guess)
        if a == 1.0:
            correct += 1.0
    total = float(len(x))
    accuracy = (correct / total) * 100.0
    print("C: ", correct, "ACCURACY: ", accuracy)

print("AFTER TRAIN EVAL")
evalute_output(best, test,lbtest)


        
l = [1,2,3]
print(len(l) % 1)