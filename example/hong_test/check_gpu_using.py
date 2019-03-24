'''
https://lucycle.tistory.com/237
tensorflow가 CPU 에서 동작하는지 확인

무조건 에러가 나서 제대로 동작하지 않는거 같다.
'''

import sys
import numpy as np
import tensorflow as tf
from datetime import datetime


def runWithGpu():
    shape=(int(10000),int(10000))

    with tf.device("/gpu:0"):
        random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
        dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
        sum_operation = tf.reduce_sum(dot_operation)

    startTime = datetime.now()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
        result = session.run(sum_operation)
        print(result)

    print("\n" * 2)
    print("Time taken:", datetime.now() - startTime)
    print("\n" * 2)

def runWithoutGpu():
    shape=(int(10000),int(10000))

    with tf.device("/cpu:0"):
        random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
        dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
        sum_operation = tf.reduce_sum(dot_operation)

    startTime = datetime.now()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
        result = session.run(sum_operation)
        print(result)

    print("\n" * 2)
    print("Time taken:", datetime.now() - startTime)
    print("\n" * 2)


def testAnother():
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

# runWithoutGpu()
runWithGpu()
