import numpy as np
import Neuron

data = np.loadtxt("iris.data", delimiter=',')

featuresA = data[0:50, 0:-1]
featuresB = data[50:100, 0:-1]
featuresC = data[100:150, 0:-1]

classes = [[0, 0], [0, 1], [1, 0]]

N = Neuron.Agent(4, 3, 2, 0.2, 0.5)
# Train

i = 0

while i < 1000:
    j = 0
    while j < 50:
        O, H = N.calcNetOutput(list(featuresA[j]), True)
        N.trainingEpisode(classes[0], O, H, list(featuresA[j]))

        O, H = N.calcNetOutput(list(featuresB[j]), True)
        N.trainingEpisode(classes[1], O, H, list(featuresB[j]))

        O, H = N.calcNetOutput(list(featuresC[j]), True)
        N.trainingEpisode(classes[2], O, H, list(featuresC[j]))

        j += 1
    i += 1

O = N.calcNetOutput(list(featuresA[0]), False)
print(O)

O = N.calcNetOutput(list(featuresB[0]), False)
print(O)

O = N.calcNetOutput(list(featuresC[0]), False)
print(O)