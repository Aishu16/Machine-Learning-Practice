import sys
import random
import math

##read data file
datafile = sys.argv[1]
file = open(datafile)

# file=open("climate.data")
data = []
i = 0
dataread = file.readline()

while (dataread != ''):  # read
    datasplit = dataread.split()
    datalength = len(datasplit)
    tempdata = []
    for j in range(0, datalength, 1):
        tempdata.append(float(datasplit[j]))
        if j == (datalength - 1):
            tempdata.append(float(1))
    data.append(tempdata)
    dataread = file.readline()

rows = len(data)
cols = len(data[0])

file.close()

##read label data
traindatafile = sys.argv[2]
file = open(traindatafile)

# file=open("climate.trainlabels.0")
trainlabels = {}

noOfItems = []
noOfItems.append(0)
noOfItems.append(0)

dataread = file.readline()

while (dataread != ''):  # read

    datasplit = dataread.split()
    trainlabels[int(datasplit[1])] = int(datasplit[0])
    dataread = file.readline()
    noOfItems[int(datasplit[0])] += 1

##initialize w

w = []
for j in range(cols):
    w.append(0)
    w[j] = (0.02 * random.uniform(0, 1)) - 0.01


#    w[j] = 1
##define function dot_product

def dotproduct(list1, list2):
    dotproduct = 0
    refw = list1
    refx = list2
    for j in range(cols):
        dotproduct += refw[j] * refx[j]

    return dotproduct


##gradient descent iteration

eta = 0.001

##calculate error outside the loop

error = 0.0

for i in range(rows):
    if (trainlabels.get(i) != None):
        d_p = dotproduct(w, data[i])
        # error += math.log(1 + math.exp((-1 * (trainlabels.get(i))) * (dotproduct(w, data[i]))))
        sigmoid = (1 / (1 + math.exp(-1 * d_p)))
        error += (-trainlabels.get(i) * math.log(sigmoid)) - ((1 - trainlabels.get(i)) * math.log(1 - sigmoid))

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
            d_p = dotproduct(w, data[i])
            # expo = (trainlabels.get(i)) - (1 / (1 + (math.exp(-1 * d_p))))
            for j in range(cols):
                # if(d_p*trainlabels.get(i)<1):
                # delf[j]+=(expo) * data[i][j]
                delf[j] += (-trainlabels.get(i) + (1 / (1 + math.exp(-d_p)))) * data[i][j]
            # delf[j]+=-1*data[i][j]*trainlabels.get(i)

            # else:
            #   delf[j]+=0
    ##update
    for j in range(cols):
        w[j] = w[j] - eta * delf[j]

    ##compute error
    curr_error = 0
    for i in range(rows):
        if (trainlabels.get(i) != None):
            d_p = dotproduct(w, data[i])
            sigmoid = (1 / (1 + math.exp(-1 * d_p)))
            curr_error += (-trainlabels.get(i) * math.log(sigmoid)) - ((1 - trainlabels.get(i)) * math.log(1 - sigmoid))
            # curr_error += math.log(1 + math.exp((-1 * (trainlabels.get(i))) * (dotproduct(w, data[i]))))
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

        d_p = dotproduct(w, data[i])

        if (d_p > 0.5):

            print("1", i)

        else:

            print("0", i)
