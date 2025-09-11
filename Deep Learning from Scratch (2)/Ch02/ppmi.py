import sys
sys.path.append('..')   # 부모 디텍터리의 파일을 가져오도록 설정
import numpy as np
from common.util import preprocess, create_co_matrix, ppmi

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

np.set_printoptions(precision=3)    # 유효 자릿수를 세 자리로 표시
print('동시발생 형렬')
print(C)
print('-'*50)
print('PPMI')
print(W)