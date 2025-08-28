import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train),(x_test, t_test) = \
      load_mnist(normalize = True, one_hot_label = True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch= x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)   # 수지적 미분값
grad_backdrop = network.gradient(x_batch, t_batch)              # 오차역전파법의 해석적 미분값

# 각 가중치의 차이의 절댓값을 구한 후, 그 절댓값들의 평균을 낸다.
# 수치적 미분과 해석적 미분의 각 구현의 결과를 비교하여 그 차이가 매우 작음을 확인하고 오차역전파법 구현에 오류가 없는지 검증할 수 있다.
for key in grad_numerical.keys():
    diff  = np.average(np.abs(grad_backdrop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))