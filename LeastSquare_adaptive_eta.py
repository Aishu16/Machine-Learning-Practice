import sys
import random

##read data file
file = sys.argv[1]
# datafile = open("ionosphere.data")
datafile = open(file)
data = []
i = 0
dataread = datafile.readline()

while (dataread != ''):
    a = dataread.split()
    b = len(a)
    tempdata = []
    for j in range(0, b, 1):
        tempdata.append(float(a[j]))
        if j == (b - 1):
            tempdata.append(float(1))

    data.append(tempdata)
    dataread = datafile.readline()

rows = len(data)
cols = len(data[0])

datafile.close()
# --------------------------------------------------------------
##read label data
file = sys.argv[2]
# datafile = open("ionosphere.trainlabels.0")
datafile = open(file)

trainlabels = {}
noOfItem = []
noOfItem.append(0)
noOfItem.append(0)

dataread = datafile.readline()
while (dataread != ''):  # read

    a = dataread.split()
    if int(a[0]) == 0:
        trainlabels[int(a[1])] = -1
    else:
        trainlabels[int(a[1])] = int(a[0])
    dataread = datafile.readline()
    noOfItem[int(a[0])] += 1

##initialize w
w = []

for j in range(cols):
    w.append(0)
    w[j] = (0.02 * random.uniform(0, 1)) - 0.01


##define function dot_product
def dp(list1, list2):
    dp = 0
    refw = list1
    refx = list2
    for j in range(cols):
        dp += refw[j] * refx[j]
    return dp


##gradient descent iteration

# initialize flag and iteration parameters
flag = 0
k = 0

while (flag != 1):
    k += 1
    delf = []
    for i in range(cols):
        delf.append(0)

    for i in range(rows):
        if (trainlabels.get(i) != None):
            dot_product = dp(w, data[i])
            for j in range(cols):
                # compute gradient
                delf[j] += (-trainlabels.get(i) + dot_product) * data[i][j]

    # choose best eta
    eta_list = [1, .1, .01, .001, .0001, .00001, .000001, .0000001, .00000001, .000000001, .0000000001, .00000000001]
    obj = 0.0
    bestobj = 1000000000000
    for k in range(0, len(eta_list), 1):
        eta = eta_list[k]
        for j in range(0, cols, 1):
            w[j] = w[j] - eta * delf[j]
        error = 0.0
        for i in range(0, rows, 1):
            if (trainlabels.get(i) != None):
                error += (-trainlabels.get(i) + dp(w, data[i])) ** 2
        obj = error
        if (obj < bestobj):
            best_eta = eta
            bestobj = obj

        for j in range(0, cols, 1):
            w[j] = w[j] + eta * delf[j]

        ##print("best eta = ",best_eta)

    ##update
    eta = best_eta
    for j in range(cols):
        w[j] = w[j] - eta * delf[j]
    print("best eta = ", best_eta)
    ##compute error
    curr_error = 0
    for i in range(rows):
        if (trainlabels.get(i) != None):
            curr_error += (-trainlabels.get(i) + dp(w, data[i])) ** 2
    print(error, k)
    if error - curr_error < 0.001:
        flag = 1
    error = curr_error

print("w =", w[:-1])

normw = 0
for j in range((cols - 1)):
    normw += w[j] ** 2
    # print(w[j])

normw = (normw) ** 0.5
print("||w||=", normw)

origin_distance = w[(len(w) - 1)] / normw
print("origin_distance =", abs(origin_distance))

for i in range(rows):
    if (trainlabels.get(i) == None):
        dot_product = dp(w, data[i])
        if (dot_product > 0):
            print("1", i)
        else:
            print("0", i)




