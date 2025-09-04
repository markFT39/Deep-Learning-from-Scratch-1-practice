import numpy as np
import gradient_1d as grad_1d

def numerical_gradient(f, x):
    h = 1e-4    # 0.0001
    grad = np.zeros_like(x)     # x와 같은 형상의 배열 생성

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val    # 값 복원

    return grad

if __name__ == '__main__':
    print(numerical_gradient(grad_1d.function_2, np.array([3.0, 4.0])))
    print(numerical_gradient(grad_1d.function_2, np.array([0.0, 2.0])))
    print(numerical_gradient(grad_1d.function_2, np.array([3.0, 0.0])))