'''
https://pinkwink.kr/1122?category=580892
'''
import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import tensorflow as tf
import keras

print('Python version : ', sys.version)
print('TensorFlow version : ', tf.__version__)
print('Keras version : ', keras.__version__)

from keras.models import load_model
model = load_model('mnist_mlp_model.h5')

# model.summary()


import matplotlib.pyplot as plt

# test1 = plt.imread('./7.png')
# plt.imshow(test1);
# print("== 읽어들임. 이미지")
# test1 = plt.imread('./7.jpg')
# plt.imshow(test1);

def printTrainData(arr, idx):
    for x in arr[idx]:
        for i in x:
            # print('{:3} '.format(i), end='')
            printNum(i)
        print()

# print(Y_train.shape)

def printNum(n):
    if(n>0):
#         print('{:3} '.format(n), end='')
        print(' ' ,n, end='')
    else:
        print('    ', end='')

test_num = plt.imread('7.png')
print("test_num.shape : ", test_num.shape)
test_num = test_num[:,:,0]
print("test_num.shape : " , test_num.shape)
# printTrainData(test_num, 0)
# test_num = (test_num > 125) * test_num
test_num = test_num.astype('float32') / 255.

# plt.figure()
plt.imshow(test_num, cmap='Greys', interpolation='nearest');
# plt.imshow(test_num, interpolation='nearest');
plt.show()

test_num = test_num.reshape((1, 28, 28, 1))
print("test_num.shape-3: " , test_num.shape)
test_num = 1-test_num
printTrainData(test_num, 0)
print("test_num.shape-10 : ", test_num.shape)
 
print('The Answer is ', model.predict_classes(test_num))

