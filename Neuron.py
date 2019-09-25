import random
import math


class Agent:
    def __init__(self, numInputs, numHidden, numOutputs,
                 initWeightSD, learningRate):

        ## we add 1 to the number of inputs here for the bias node
        numInputs += 1

        self.inputToHidden = []
        self.hiddenToOutput = []
        self.numInputs = numInputs
        self.numHidden = numHidden
        self.numOutputs = numOutputs
        self.learningRate = learningRate

        for i in range(numInputs):
            workingList = []
            for h in range(numHidden):
                newWeight = random.normalvariate(0.0, initWeightSD)
                workingList.append(newWeight)
            self.inputToHidden.append(workingList)

        for h in range(numHidden + 1):  ## +1 to cover the bias-to-output connections
            workingList = []
            for o in range(numOutputs):
                newWeight = random.normalvariate(0.0, initWeightSD)
                workingList.append(newWeight)
            self.hiddenToOutput.append(workingList)

    def printNetwork(self, name):
        print(name, "network weights")
        for i in range(self.numInputs):
            for j in range(self.numHidden):
                print(self.inputToHidden[i][j]),
            print()
        print("--------------------")
        for j in range(self.numHidden):
            for k in range(self.numOutputs):
                print(self.hiddenToOutput[j][k]),
            print()

    def calcNetOutput(self, inputToNet, wantHiddenLevels):
        """Input is list of real-valued inputs.  Also a Boolean specifying
        whether you'd like the hidden layer activation levels returned or not.
        Output is a list of real-valued outputs, and optionally the hidden layer
        activation levels too.  Both are sigmoid-functioned before being returned.
        Note that if you get confused about whether you want the hidden layer
        levels or not, things can get messed up, as you'll get two lists returned
        instead of one."""

        ## immediately add 1.0 to the input list: this is the bias node
        inputWithBias = inputToNet[:]
        inputWithBias.append(1.0)

        hiddenActivationLevels = [0.0] * self.numHidden
        outputActivationLevels = [0.0] * self.numOutputs

        for i in range(self.numInputs):
            for h in range(self.numHidden):
                hiddenActivationLevels[h] += (inputWithBias[i]
                                              * self.inputToHidden[i][h])

        for h in range(self.numHidden):
            hiddenActivationLevels[h] = self.sigmoid(hiddenActivationLevels[h])

        hiddenActivationLevels.append(1.0)  ## this is the bias-to-output node

        for h in range(self.numHidden + 1):  ## +1 to cover the bias-to-output connections
            for o in range(self.numOutputs):
                outputActivationLevels[o] += (hiddenActivationLevels[h]
                                              * self.hiddenToOutput[h][o])

        ## Note that we sigmoid the output functions before we send them back
        ## This is because they need to be in the 0--1 range so we can compare
        ## them to bit strings and judge their closeness.
        for o in range(self.numOutputs):
            outputActivationLevels[o] = self.sigmoid(outputActivationLevels[o])

        if wantHiddenLevels:
            return outputActivationLevels, hiddenActivationLevels
        else:
            return outputActivationLevels

    def sigmoid(self, before):
        """Returns the classic neural-net sigmoid of the `before' value.
        So the return value is bounded by 0 and 1, but the input value
        can be anything (pos or neg, bounded by infiinity)."""
        retValue = 1.0 / (1.0 + math.exp(0.0 - before))
        return retValue

    def trainingEpisode(self, targetOutput, actualOutput,
                        hiddenOutput, actualInput):
        """This is where the backpropagation maths get implemented.
        We do one training event where the weights of the network
        get properly updated for a given target output and a given
        actual output.  Both of these are lists of real values."""

        # as before, add 1.0 to the input list to implement the bias node
        inputWithBias = actualInput[:]
        inputWithBias.append(1.0)

        hiddenWithBias = hiddenOutput[:]
        hiddenWithBias.append(1.0)

        # calculate deltaKs: basically an error measure per output neuron
        deltaK = [(t - y) * y * (1 - y)
                  for t, y in zip(targetOutput, actualOutput)]

        # print "deltaK = ", deltaK

        # train the hidden-to-output connections
        for j in range(self.numHidden + 1):  ## +1 for the bias-to-output connections
            # print "HO", j, "=", hiddenWithBias[j]
            for k in range(self.numOutputs):
                deltaW = self.learningRate * deltaK[k] * hiddenWithBias[j]
                self.hiddenToOutput[j][k] += deltaW
                # print "H->O weight delta", j, k, deltaW

        # calculate deltaJs: basically an error measure for hidden neurons
        deltaJ = [0.0] * self.numHidden
        for j in range(self.numHidden):  ## deliberate use of numHidden, not numHidden + 1
            sigma = 0.0
            for k in range(self.numOutputs):
                sigma += self.hiddenToOutput[j][k] * deltaK[k]
            deltaJ[j] = hiddenOutput[j] * (1 - hiddenOutput[j]) * sigma

        # print "deltaJ = ", deltaJ

        # train the input-to-hidden connections
        for i in range(self.numInputs):
            # print "AI", i, "=", inputWithBias[i]
            for j in range(self.numHidden):
                deltaW = self.learningRate * deltaJ[j] * inputWithBias[i]
                self.inputToHidden[i][j] += deltaW
                # print "I->H weight delta", i, j, deltaW

    def produce(self, bitStrings, meaningSelecter):
        best = float(0)
        for st in range(len(bitStrings)):
            inputToNet = bitStrings[st].floatRep
            outputFromNet = self.calcNetOutput(inputToNet, False)
            similarity = self.similarity(outputFromNet, bitStrings, meaningSelecter)
            if similarity > best:
                best = similarity
                s = st
        return s

    def similarity(self, outputFromNet, bitStrings, meaningSelecter):

        netScore = 1.0
        for i in range(len(bitStrings[1].floatRep)):  # bitStrings[i])):
            goal = bitStrings[meaningSelecter].floatRep[i]
            attempt = outputFromNet[i]
            if goal == 1.0:
                netScore *= attempt  # delta = attempt
            else:  # if goal == 0.0:
                netScore *= 1.0 - attempt

            # netScore *= delta

        return netScore

    def produceStability(self, bitStrings, index):
        best = float(0)
        for st in range(len(bitStrings)):
            inputToNet = bitStrings[st].floatRep
            outputFromNet = self.calcNetOutput(inputToNet, False)
            similarity = self.similarity(outputFromNet, bitStrings, index)
            if similarity > best:
                best = similarity
                s = st
        return s