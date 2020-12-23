import tensorflow as tf
import random 
import numpy as np

l = 640
a = np.arange(l)
a1 = np.arange(l)
a2 = np.random.permutation(l)
b = np.random.permutation(l)
print(a)
print(b)

i = 0
j = b[i]
c1 = [i]


while(j != c1[0]):
    i = j
    j = b[i]
    #a1.remove()
    c1.append(i)

c1 = sorted(c1)
print(c1)
        
w = []

i = 0
j = 0
while(i < len(c1)):
    v = c1[i]
    x = a1[v]
    print(i, v, x)
    v2 = v + 1
    i = i + 1
    w.extend([x])
    if i < len(c1):
        v3 = c1[i]
        y = a2[v2:v3]
        w.extend(y)
    else:
        if v < len(a1) - 1:
            w.extend(a2[v+1:])

        

print(len(w))
    
