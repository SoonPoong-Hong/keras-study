'''
http://snowdeer.github.io/machine-learning/2018/01/09/recognize-mnist-data/

'''

from keras.datasets import mnist

(X_train, Y_train), (X_validation, Y_validation) = mnist.load_data()

print ("x_train shape : " , X_train.shape)

def printTrainData(idx):
    for x in X_train[idx]:
        for i in x:
            # print('{:3} '.format(i), end='')
            printNum(i)
        print()

# print(Y_train.shape)

def printNum(n):
    if(n>0):
        print('{:3} '.format(n), end='')
    else:
        print('    ', end='')

def printYTrain(idx):
    print("Y_train", idx, " => ", Y_train[idx])

printTrainData(2)
print("========================")
printYTrain(2)
