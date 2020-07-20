import sys
import math
import random

# Read files from the command line
# datafile = sys.argv[1]
#f = open("ionosphere.data")

datafile = sys.argv[1]
f = open(datafile, 'r')
data = []
# i = 0
l = f.readline()

# Read Data
while (l != ''):
    a = l.split()
    l2 = []
    for j in range(0, len(a), 1):
        l2.append(float(a[j]))
    data.append(l2)
    # print(data)
    l = f.readline()

rows = len(data)
cols = len(data[0])
print("rows:", rows, " ", "columns:", cols)

# Read Training Labels
# trainlabelfile = sys.argv[2]

#f = open("ionosphere.labels")


trainablefile = sys.argv[2]
f = open(trainablefile, 'r')
trainlabels = {}
n = [0, 0]
l = f.readline()
while (l != ''):
    a = l.split()
    # print("a = ",a)
    trainlabels[int(a[1])] = int(a[0])

    '''if(trainlabels[int(a[1])] == 0):
        trainlabels[int(a[1])] = -1'''

    l = f.readline()
    n[int(a[0])] += 1
# trows = len(trainlabels)
# print(data)
# print("trainlabels",trainlabels)
print("================================================")


# Define classifier with gini index
def giniClassifier(data, labels, col):
    colvals = {}
    indices = []
    rows = 0
    minus = 0
    for i in range(0, len(data), 1):
        if (labels.get(i) != None):
            colvals[i] = data[i][col]
            indices.append(i)
            rows += 1
            if (labels[i] == 0):
                minus += 1
    # sorted_indices = sorted(indices, key= lambda k: colvals[k])
    sorted_indices = sorted(indices, key=colvals.__getitem__)

    lsize = 1
    rsize = rows - 1
    lp = 0
    rp = minus
    if (labels[sorted_indices[0]] == 0):
        lp += 1
        rp -= 1

    best_s = -1
    bestgini = 10000
    for i in range(1, len(sorted_indices), 1):
        s = (colvals[sorted_indices[i]] + colvals[sorted_indices[i - 1]]) / 2
        gini = (lsize / rows) * (lp / lsize) * (1 - lp / lsize) + (rsize / rows) * (rp / rsize) * (1 - rp / rsize)
        # print(lp,lsize,rp,rsize)
        if (gini < bestgini):
            bestgini = gini
            best_s = s
        if (labels[sorted_indices[i]] == 0):
            lp += 1
            rp -= 1
        lsize += 1
        rsize -= 1
    return (best_s, bestgini)


# Main Body

best_split = -1
best_col = -1
best_gini = 10000
for j in range(0, cols, 1):
    [s, gini] = giniClassifier(data, trainlabels, j)
    # print(s,gini)
    if (gini < best_gini):
        best_gini = gini
        best_split = s
        best_col = j
n = 0
p = 0
for i in range(0, rows, 1):
    if (trainlabels.get(i) != None):
        if (data[i][best_col] < best_split):
            if (trainlabels[i] == 0):
                n += 1
            else:
                p += 1

if (n > p):
    left = 0
    right = 1
else:
    left = 1
    right = 0
print("Gini is =", best_gini, "Split point value=", best_split, "Best_Column=", best_col)

# Classify unlabeled points
for i in range(0, rows, 1):
    if (trainlabels.get(i) == None):
        if (data[i][best_col] < best_split):
            print(left, i)
        else:
            print(right, i)
