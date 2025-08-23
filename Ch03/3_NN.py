import numpy as np
import sigmoid as sig

# 0층(입력층) 정의
X = np.array([1.0, 0.5])

# ------------------------------------------------------------ #

# 1층(은닉층) 계산
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape) # (2, 3)
print(X.shape)  # (2, )
print(B1.shape) # (3, )

# 입력 신호 총합
A1 = np.dot(X, W1) + B1
# 활성화 함수 및 출력값 (시그모이드)
Z1 = sig.sigmoid(A1)

print(A1)       # [0.3 0.7 1.1]
print(Z1)       # [0.57444252 0.66818777 0.75026011]

print("End of 1st Layer!\n")

# ------------------------------------------------------------ #

# 2층(은닉층) 계산
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape) # (3, )
print(W2.shape) # (3, 2)
print(B2.shape) # (2, )

# 입력 신호 총합
A2 = np.dot(Z1, W2) + B2
# 활성화 함수 및 출력값 (시그모이드)
Z2 = sig.sigmoid(A2)

print(A2)       # [0.51615984 1.21402696]
print(Z2)       # [0.62624937 0.7710107 ]

print("End of 2nd Layer!\n")

# ------------------------------------------------------------ #

# 3층(출력층) 출력
def indentify_function(x):
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

print(Z2.shape) # (2, )
print(W3.shape) # (2, 2)
print(B3.shape) # (2, )

A3 = np.dot(Z2, W3) + B3
Y = indentify_function(A3)  # Y == A3

print(A3)       # [0.31682708 0.69627909]
print(Y)        # 

print("End of last Layer!\n")

print("Result: ")
print(Y)