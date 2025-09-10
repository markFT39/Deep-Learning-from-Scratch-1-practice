import sys
sys.path.append('..')   # 부모 디텍터리의 파일을 가져오도록 설정
from common.util import preprocess, create_co_matrix, cos_similarity, most_similar

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

c0 = C[word_to_id['you']]   # "you"의 단어 벡터
c1 = C[word_to_id['i']]     # "i"의 단어 벡터
print(cos_similarity(c0, c1))       # 0.7071067691154799

most_similar('you', word_to_id, id_to_word, C, top=5)