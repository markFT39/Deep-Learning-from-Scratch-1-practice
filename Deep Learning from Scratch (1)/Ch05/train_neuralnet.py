import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pylab as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train),(x_test, t_test) = \
      load_mnist(one_hot_label= True, normalize= True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 하이퍼 파라미터
iters_num = 10000   # 반복횟수
train_size = x_train.shape[0]
batch_size = 100    # 미니 배치 크기
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에포크당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # 미니 배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    # grad = network.numerical_gradient(x_batch, t_batch)     # 계산이 오래걸려서 샐행 중단, 30분~1시간 예상됨...
    grad = network.gradient(x_batch, t_batch)     # 성능 개선, 오차역전파법 사용

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1에포크당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 손실 함수 그래프 그리기
plt.plot(train_loss_list)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.title("Training Loss")
plt.show()

# 정확도 그래프 그리기
epochs = np.arange(len(train_acc_list))
plt.plot(epochs, train_acc_list, label='train acc')
plt.plot(epochs, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend()
plt.title("Training and Test Accuracy")
plt.show()