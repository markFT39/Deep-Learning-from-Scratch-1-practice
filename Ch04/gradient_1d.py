import numpy as np
import matplotlib.pylab as plt

# 수치 미분: 미분값의 근사치 계산법
# 해석적 미분: 오차를 포함하지 않는 계산법

# 수치 미분법 중 중심 차분 or 중앙 차분
def numerical_diff(f, x):
    h = 1e-4    # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

# y = 0.01x^2 + 0.1x
# y' = 0.02x + 0.1
def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x

# f(x1, x2) = x0^2 + x1^2
def function_2(x):
    # return np.sum(x ** 2)
    return x[0] ** 2 + x[1] ** 2


if __name__ == '__main__':
    x = np.arange(0.0, 20.0, 0.1)
    y = function_1(x)

    print(numerical_diff(function_1, 5))    # 수치 미분 값: 0.1999999999990898, 해석적 미분 값: 0.2
    print(numerical_diff(function_1, 10))   # 수치 미분 값: 0.2999999999986347, 해석적 미분 값: 0.3

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, y)
    plt.show()

    