import sys
import random

##read data file
file = sys.argv[1]
# datafile = open("ionosphere.data")
datafile = open(file)
dataread = datafile.readline()
data = []
i = 0

while (dataread != ''):  # read

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
dataread = datafile.readline()
trainlabels = {}

noOfItems = []
noOfItems.append(0)
noOfItems.append(0)

while (dataread != ''):  # read

    a = dataread.split()

    if int(a[0]) == 0:
        trainlabels[int(a[1])] = -1

    else:

        trainlabels[int(a[1])] = int(a[0])
    dataread = datafile.readline()
    noOfItems[int(a[0])] += 1

##initialize w

w = []
for j in range(cols):
    w.append(0)
    w[j] = (0.02 * random.uniform(0, 1)) - 0.01


#    w[j] = 1
##define function dot_product

def dp(list1, list2):
    dp = 0
    refw = list1
    refx = list2
    for j in range(cols):
        dp += refw[j] * refx[j]

    return dp


##gradient descent iteration
##calculate error outside the loop

error = 0.0

for i in range(rows):

    if (trainlabels.get(i) != None):
        error += max(0, 1 - trainlabels.get(i) * dp(w, data[i]))

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
            d_p = dp(w, data[i])

            for j in range(cols):
                if (d_p * trainlabels.get(i) < 1):
                    delf[j] += -1 * data[i][j] * trainlabels.get(i)

                else:

                    delf[j] += 0
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
                error += max(0, 1 - trainlabels.get(i) * dp(w, data[i]))
        obj = error
        if (obj < bestobj):
            best_eta = eta
            bestobj = obj

        for j in range(0, cols, 1):
            w[j] = w[j] + eta * delf[j]
        ##update
    print("best eta = ", best_eta)
    eta = best_eta
    for j in range(cols):
        w[j] = w[j] - eta * delf[j]

    ##compute error

    curr_error = 0

    for i in range(rows):

        if (trainlabels.get(i) != None):
            curr_error += max(0, 1 - trainlabels.get(i) * dp(w, data[i]))

    print(error, k)

    if error - curr_error < 0.001:
        flag = 1

    error = curr_error

normw = 0
for j in range((cols - 1)):
    normw += w[j] ** 2

    print(w[j])

normw = (normw) ** 0.5

print("||w||=", normw)
distance_origin = w[(len(w) - 1)] / normw
print(distance_origin)

for i in range(rows):

    if (trainlabels.get(i) == None):

        d_p = dp(w, data[i])

        if (d_p > 0):

            print("1", i)

        else:

            print("0", i)
