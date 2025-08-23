import numpy as np
import matplotlib.pyplot as plt

# 데이터 준비
x = np.arange(0, 6, 0.1) # 0부터 6까지 0.1 간격
# x = [0, 0.1, 0.2, 0.3, ... , 5.8, 5.9]
y = np.sin(x)

# 그래프 그리기
plt.plot(x, y)
plt.show()