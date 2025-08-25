import numpy as np
import gradient_1d as grad_1d
import gradient_2d as grad_2d

def gradient_desent(f, init_x, lr = 0.01, step_num = 100):
    # lr: 학습률, step_num: 반복 횟수
    x = init_x

    for i in range(step_num):
        grad = grad_2d.numerical_gradient(f, x)
        x -= lr * grad
    
    return x

if __name__ == '__main__':
    init_x = np.array([-3.0, 4.0])
    print(gradient_desent(grad_1d.function_2, init_x=init_x, lr=0.1, step_num=100))

    # 학습률이 너무 큰 경우, lr = 10.0
    init_x = np.array([-3.0, 4.0])
    print(gradient_desent(grad_1d.function_2, init_x=init_x, lr=10.0, step_num=100))

    # 학습률이 너무 작은 경우, lr = 1e-10
    init_x = np.array([-3.0, 4.0])
    print(gradient_desent(grad_1d.function_2, init_x=init_x, lr=1e-10, step_num=100))