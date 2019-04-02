'''
https://pinkwink.kr/1122?category=580892
'''
# 0. 사용할 패키지 불러오기
import sys
import tensorflow as tf
import keras

from keras.models import load_model
model = load_model('mnist_mlp_model.h5')

model.summary()
