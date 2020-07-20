import sys
from sys import argv
import random
import math
from collections import Counter

#################
### Read Data ###
#################
file = sys.argv[1]
datafile = open(file)
data = []
dataread = datafile.readline()
while (dataread != ''):
    a = dataread.split()
    l2 = []
    for j in range(0, len(a), 1):
        l2.append(float(a[j]))
    data.append(l2)
    dataread = datafile.readline()
rows = len(data)
cols = len(data[0])
datafile.close()

###############################
##### read training labels ####
###############################

file = sys.argv[2]
datafile = open(file)
trainlabels = {}
dataread = datafile.readline()
numclass = []
numclass.append(0)
numclass.append(0)
while (dataread != ''):
    a = dataread.split()
    trainlabels[int(a[1])] = int(a[0])
    numclass[int(a[0])] = numclass[int(a[0])] + 1
    dataread = datafile.readline()
datafile.close()


def bootstrap(trainlabels, data, rows, cols):
    newkey = list()
    sampledata = []
    sampletrainlabels = {}
    for i in trainlabels.keys():
        newkey.append(i)
    # while (i != trainlabels.keys())
    #    newkey.append(i)

    # print(len(newkey))
    for i in range(0, len(newkey), 1):
        rowselect = random.choice(newkey)
        sampledata.append(data[rowselect])
        sampletrainlabels[i] = trainlabels.get(rowselect)
    datarows = len(sampledata)
    datacols = len(sampledata[0])
    return (sampledata, sampletrainlabels, datarows, datacols)


def ginical(data, trainlabels, rows, cols, pridict_val, row_ori, col_ori, mtrainlabels, mdata):
    ginivals = []
    split = 0
    l3 = [0, 0]
    for j in range(0, cols, 1):
        ginivals.append(l3)
    temp = 0
    col = 0

    for j in range(0, cols, 1):

        listcol = [item[j] for item in data]
        keys = sorted(range(len(listcol)), key=lambda k: listcol[k])
        listcol.sort()
        ginival = []
        prevgini = 0
        prevrow = 0
        for k in range(1, rows, 1):

            lsize = k
            rsize = rows - k
            lp = 0
            rp = 0

            for l in range(0, k, 1):
                if (trainlabels.get(keys[l]) == 0):
                    lp += 1

            for r in range(k, rows, 1):
                if (trainlabels.get(keys[r]) == 0):
                    rp += 1
            gini = (lsize / rows) * (lp / lsize) * (1 - lp / lsize) + (rsize / rows) * (rp / rsize) * (1 - rp / rsize)
            # print(gini)
            ginival.append(gini)

            prevgini = min(ginival)
            if (ginival[k - 1] == float(prevgini)):
                ginivals[j][0] = ginival[k - 1]
                ginivals[j][1] = k

        if (j == 0):
            temp = ginivals[j][0]
            # print("temp",temp)
        if (ginivals[j][0] <= temp):
            temp = ginivals[j][0]
            col = j
            split = ginivals[j][1]
            # print("split",split)
            if (split != 0):
                split = (listcol[split] + listcol[split - 1]) / 2

    variable = 0
    m = 0
    p = 0
    for i in range(0, row_ori, 1):
        if (mtrainlabels.get(i) != None):
            if (mdata[i][col] < split):
                if (mtrainlabels.get(i) == 0):
                    m += 1
                if (mtrainlabels.get(i) == 1):
                    p += 1
    if (m > p):
        left = 0
        right = 1
    else:
        left = 1
        right = 0

    for i in range(0, row_ori, 1):
        if (mtrainlabels.get(i) == None):
            if (mdata[i][col] < split):
                variable = left
            else:
                variable = right

            if i in pridict_val:
                pridict_val[i].append(variable)
            else:
                pridict_val[i] = [variable]

    return (pridict_val)


pridict_val = dict()
row_ori = len(data)
col_ori = len(data)
mdata = data
mtrainlabels = trainlabels
for i in range(0, 5, 1):
    # sampledata,sampletrainlabels,datarows,datacols = bootstrap(trainlabels,data)
    sampledata, sampletrainlabels, datarows, datacols = bootstrap(trainlabels, data, rows, cols)
    # pridict_val = ginical(trainlabels,data,datarows,datacols,pridict_val)
    pridict_val = ginical(sampledata, sampletrainlabels, datarows, datacols, pridict_val, row_ori, col_ori,
                          mtrainlabels, mdata)
    print(len(pridict_val))

for k, v in pridict_val.items():
    count = Counter(pridict_val[k])
    a = count.most_common()[0]
    print(a[0], k)
