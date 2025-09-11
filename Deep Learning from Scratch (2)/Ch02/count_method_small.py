import sys
sys.path.append('..')   # 부모 디텍터리의 파일을 가져오도록 설정
import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, ppmi

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

# SVD
U, S, V = np.linalg.svd(W)

print(C[0]) # 동시발생 행렬
# [0 1 0 0 0 0 0]

print(W[0]) # PPMI 행렬
# [0.        1.8073549 0.        0.        0.        0.        0.       ]

print(U[0]) # SVD
# [-1.1102230e-16  3.4094876e-01 -1.2051624e-01 -3.8857806e-16
#  0.0000000e+00 -9.3232495e-01  8.7683712e-17]

print(U[0, :2])
# [-1.1102230e-16  3.4094876e-01]

for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))

plt.scatter(U[:,0], U[:,1], alpha=0.5)
plt.show()