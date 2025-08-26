import numpy as np

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <=0)     # 배열 x에 대해 각 원소가 0 이하이면 True 그 외엔 False를 갖는 배열 mask를 생성함
        out = x.copy()          # x의 복사로 원본 x 저장
        out[self.mask] = 0      # out에서 mask가 True인 위치만을 0으로 변경

        return out
    
    def backward(self, dout):
        dout[self.mask] = 0     # 순전파때 0이었던 값들만을 0으로 변경해 전달함
        dx = dout

        return dx
    
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out
    
    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)   # Sigmoid 역전파 값은 dL/dy * y * (1 - y)로 출력 y로만 계산 가능하다

        return dx