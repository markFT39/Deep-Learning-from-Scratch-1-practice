import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7    # y가 0일 경우 log0 = -inf로 계산 불가. 따라서 매우 작은 값을 더해 log0이 발생하지 않도록 한다
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta)) / batch_size
    # t가 one-got encoding이 아닐 때, 정답이 숫자 레이블로 바로 주어질 때
    # return -np.sum(np.log(y[np.arragne(batch_size), t] + delta)) / batch_size

if __name__ == '__main__':
    t = [0,0,1,0,0,0,0,0,0,0]
    y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
    print(cross_entropy_error(np.array(y), np.array(t)))

    y2 = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
    print(cross_entropy_error(np.array(y2), np.array(t)))