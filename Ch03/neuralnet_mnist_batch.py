import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import sigmoid as sig
import softmax as sof
from dataset.mnist import load_mnist
import pickle

def get_data():
    (x_train, t_train),(x_test, t_test) = \
      load_mnist(normalize = True, flatten = True, one_hot_label = False)
    
    return x_test, t_test

def init_network():
    with open(os.path.dirname(__file__) + "/sample_weight.pkl", "rb") as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sig.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sig.sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = sof.softmax(a3)

    return y

x, t = get_data()
network = init_network()

batch_size = 100    # 배치 크기
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]                     # x_batch.shape = (100, 784), 2차원 배열
    y_batch = predict(network, x_batch)             # y_batch.shape = (100, 10), 2차원 배열
    p = np.argmax(y_batch, axis=1)                  # p.shape = (100,), 1차원 벡터
    accuracy_cnt += np.sum(p == t[i:i+batch_size])


print("Accuracy:" + str(float(accuracy_cnt) / len(x)))  # Accuracy:0.9352