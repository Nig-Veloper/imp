"""
# p1 
# BFS bfs : Breadth First Search

from map import dict_gn
from queue import Queue

start = "Arad"
goal = "Bucharest"
result = ""


def bfs(city, cityq, visitedq):
    global result

    if city == start:
        result += " " + city

    for eachcity in dict_gn[city].keys():

        if eachcity == goal:
            result += " " + eachcity
            return

        if eachcity not in cityq.queue and eachcity not in visitedq.queue:
            cityq.put(eachcity)
            result += " " + eachcity
    visitedq.put(city)

    bfs(cityq.get(), cityq, visitedq)


def main():
    cityq = Queue()
    visitedq = Queue()

    bfs(start, cityq, visitedq)
    print(f"BFS Traversal\n\nFrom:{start}\nTo:{goal}\n\n{result}")


main()


"""


"""
# P2 
# IDDFS , iddfs : Iterataive Deepning Depth First Search


from RMP import dict_gn

start = "Arad"
goal = "Bucharest"
result = ""


def dls(city, visitedstack, startlimit, endlimit):
    global result

    found = 0
    result += city + " "
    visitedstack.append(city)

    if city == goal:
        return 1

    if startlimit == endlimit:
        return 0

    for eachcity in dict_gn[city].keys():

        if eachcity not in visitedstack:
            found = dls(eachcity, visitedstack, startlimit+1, endlimit)

        if found:
            return found


def iddfs(city, visitedstack, endlimit):
    global result

    for i in range(0, endlimit):
        print("Searching for Limit: ", i)

        found = dls(city, visitedstack, 0, i)

        if found:
            print("Found")
            break
        else:
            print("Not Found.")
            print(result)
            result = ""
            visitedstack = []


def main():
    visitedstack = []
    iddfs(start, visitedstack, 9)
    print("Iddfs traversal from ", start, " to ", goal, " is: ", result)


main()

"""


"""
# P3
# a-star astar a* A* 

from queue import PriorityQueue
from rmp import dict_gn
from rmp import dict_hn


start = "Arad"
goal = "Bucharest"
result = ""


def expand(cityq):
    global result
    tot, citystr, thiscity = cityq.get()

    if thiscity == goal:
        result = citystr + "::" + str(tot)
        return

    for cty in dict_gn[thiscity]:
        cityq.put((get_fn(citystr+","+cty), citystr+","+cty, cty))
    expand(cityq)


def get_fn(citystr):
    cities = citystr.split(",")
    hn = gn = 0
    for ctr in range(0, len(cities)-1):
        gn = gn + dict_gn[cities[ctr]][cities[ctr+1]]
    hn = dict_hn[cities[len(cities)-1]]
    return(hn+gn)


def main():
    cityq = PriorityQueue()
    thiscity = start
    cityq.put((get_fn(start), start, thiscity))
    expand(cityq)
    print("The A* path with total is:")
    print(result)


main()

"""


"""
# P4 
# rbfs : Recursive Best First Search

import queue as q

from RPM import dict_gn, dict_hn

start = "Arad"
goal = "Bucharest"
result = ""

def get_fn(citystr):
    cities = citystr.split(",")
    hn = gn = 0
    for ctr in range(0, len(cities)-1):
        gn = gn + dict_gn[cities[ctr]][cities[ctr+1]]
    hn = dict_hn[cities[len(cities)-1]]
    return (hn+gn)


def printout(cityq):
    for i in range(0, cityq.qsize()):
        print(cityq.queue[i])

def expand(cityq):
    global result
    tot, citystr, thiscity = cityq.get()
    nexttot = 999
    if not cityq.empty():
        nexttot, nextcitystr, nextthiscity = cityq.queue[0]
    
    if thiscity == goal and tot < nexttot:
        result = citystr +":"+str(tot)
        return
    
    print("Expanded city ----", thiscity)
    print("Second best f(n) ----", nexttot)

    tempq = q.PriorityQueue()
    for cty in dict_gn[thiscity]:
        tempq.put((get_fn(citystr+","+cty),citystr+","+cty,cty))
    for ctr in range(1,3):
        ctrtot, ctrcitystr, ctrthiscity = tempq.get()
        if ctrtot < nexttot:
            cityq.put((ctrtot,ctrcitystr,ctrthiscity))
        else:
            cityq.put((ctrtot, citystr, thiscity))
            break
    
    printout(cityq)
    expand(cityq)


def main():
    cityq = q.PriorityQueue()
    thiscity = start
    cityq.put((999, "NA", "NA"))
    cityq.put((get_fn(start), start, thiscity))
    expand(cityq)
    print(result)

main()
"""


"""
# P5
# resturant decision tree 
# decision tree , decision-tree

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# func importing dataset


def importdata():
    balance_data = pd.read_csv("balance-scale.data")

    # print the dataset shape
    print("Dataset Length : ", len(balance_data))

    # printing the dataset observations
    print("Dataset : ", balance_data.head())
    return balance_data

# func to split the dataset


def splitdataset(balance_data):
    # seperating the target variable
    X = balance_data.values[:, 1:5]
    Y = balance_data.values[:, 0]

    # splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=100)
    return X, Y, X_train, X_test, y_train, y_test

# function to perform training with entropy


def train_using_entropy(X_train, X_test, y_train, y_test):
    # decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)

    # performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy


def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print("Predicted Values : ")
    print(y_pred)
    return y_pred


def cal_accuracy(y_test, y_pred):
    print("Accuracy : ", accuracy_score(y_test, y_pred)*100)


def main():
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)

    clf_entropy = train_using_entropy(X_train, X_test, y_train, y_test)

    print("Results using entropy : ")
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)


if __name__ == "__main__":
    main()

"""


"""
# P6 
# Neural Network , neural-network , neural network

import numpy as np

class NeuralNetwork():
    def __init__(self):
        np.random.seed()

        self.synaptic_weights=2*np.random.random((3,1))-1

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_derivative(self,x):
        return x*(1-x)

    def train(self,training_inputs,training_outputs,training_iterations):
        for iteration in range(training_iterations):
            output=self.think(training_inputs)
            error = training_outputs-output

            adjustments=np.dot(training_inputs.T,error*self.sigmoid_derivative(output))
            self.synaptic_weights +=adjustments
    
    def think(self,inputs):
        inputs=inputs.astype(float)
        output=self.sigmoid(np.dot(inputs,self.synaptic_weights))
        return output


if __name__ == "__main__":

    #initializing the neuron class
    neural_network = NeuralNetwork()

    print("Beginning Randomly Generated Weights: ")
    print(neural_network.synaptic_weights)

    #training data consisting of 4 examples--3 input values and 1 output
    training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1]])

    training_outputs = np.array([[0,1,1,0]]).T

    #training taking place
    neural_network.train(training_inputs, training_outputs, 15000)

    print("Ending Weights After Training: ")
    print(neural_network.synaptic_weights)

    user_input_one = str(input("User Input One: "))
    user_input_two = str(input("User Input Two: "))
    user_input_three = str(input("User Input Three: "))
    
    print("Considering New Situation: ", user_input_one, user_input_two, user_input_three)
    print("New Output data: ")
    print(neural_network.think(np.array([user_input_one, user_input_two, user_input_three])))

"""


"""
# P7 
# passive reinforcement learning algorithm

import numpy as np


def return_state_utility(v, T, u, reward, gamma):
    action_array = np.zeros(4)

    for action in range(0, 4):
        action_array[action] = np.sum(
            np.multiply(u, np.dot(v, T[:, :, action])))

    return reward + gamma * np.max(action_array)


def main():
    v = np.array(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]])

    #

    T = np.load("T.npy")

    u = np.array([[0.812, 0.868, 0.918, 1.0,
                   0.762, 0.0, 0.660, -1.0,
                   0.705, 0.655, 0.611, 0.388]])

    reward = -0.4
    gamma = 0.1
    utility = return_state_utility(v, T, u, reward, gamma)

    print(f"Utility of state : {utility}")


main()

"""


"""
# P8 
# ada_boost , ada boost , Ada Boost


import pandas
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age',
         'class']

dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]

seed = 7
num_trees = 30

model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)

results = model_selection.cross_val_score(model, X, Y)

print(results.mean())

"""
