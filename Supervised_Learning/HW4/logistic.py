#############  Logistic  ############ 

import numpy as np
import random
input_file = open("classification.txt","r")
ys = []
xs = []

for point in input_file:
    ps = map(float, point.split(","))
    xs.append([1]+ps[:3])
    ys.append(int(ps[4]))

wts = [0.0, 0.0, 0.0, 0.0]
xs = np.array(xs)
ys = np.array(ys)
wts = np.array(wts)
alp = 0.1

def deltaCalc():
    init_sum = np.array([0.0,0.0,0.0, 0.0])
    for i in range(len(xs)-1):
        init_sum += np.sign(ys[i+1])*xs[i+1]/(1+np.exp(np.sign(ys[i+1])*np.dot(xs[i+1], wts)))
    return -init_sum/len(xs)


count = 0
while count < 7000:
    count += 1
    wts += -np.dot(deltaCalc(), alp)

print "Weights %s" % wts

predYs = np.dot(xs, wts)
for i in range(len(predYs)):
    if np.exp(predYs[i])/(1 + np.exp(predYs[i])) < 0.5:
        predYs[i] = -1
    else:
        predYs[i] = 1

predYs = map(int, predYs)

count = 0
for i in range(len(predYs)):
    if predYs[i] == ys[i]:
        count += 1

f = open("logistic_regression_output.txt", "w")
f.write("weights = %s\n"% wts)
f.write("\n\n Coordinates - X, Y, Y_Predicted\n")
for i in range(len(xs)):
    f.write("%s\t%s\t%s\t%s\n"% (xs[i][1], xs[i][2], ys[i], predYs[i]))

acc = 1.0*count/len(ys)
print "Accuracy %s" % acc