# Deep Learning from Scratch - Practice

This repository contains my code implementations and learning notes based on the **"Deep Learning from Scratch"** book series by O'Reilly Japan / 사이토 고키.  
The goal is to **implement deep learning concepts from first principles** using only NumPy and Python, without relying on high-level frameworks.

📎 Related repository: [WegraLee/deep-learning-from-scratch](https://github.com/WegraLee/deep-learning-from-scratch)

📎 Related repository: [markFT39/Deep-Learning-from-Scratch-series](https://github.com/markFT39/Deep-Learning-from-Scratch-series)

---

## 📘 Book Series
- **Deep Learning from Scratch (1)**  
  Fundamentals of neural networks and deep learning using NumPy.  
  Code in: `Deep Learning from Scratch (1)/Ch01`, `Ch02`, ...

- **Deep Learning from Scratch (2)**  
  Natural Language Processing (word embeddings, RNNs, etc.).  
  Code in: `Deep Learning from Scratch (2)/Ch01`, `Ch02`, ...

- **Deep Learning from Scratch (3)**  
  Reinforcement Learning (RL) basics and implementations.  
  Code in: `Deep Learning from Scratch (3)/steps`, `dezero`, ...

- **Deep Learning from Scratch (4)** *(planned)*  
  Focus: ...

---

## 📁 Contents

### Book 1: Fundamentals
- Chapter 1: Python and NumPy Basics  
- Chapter 2: Perceptron  
- Chapter 3: Neural Networks  
- Chapter 4: Neural Network Training (loss functions, gradient methods)  
- Chapter 5: Backpropagation  
- Chapter 6: Training Techniques (weight initialization, overfitting prevention, hyperparameters)   
- Chapter 7: Convolutional Neural Networks (CNNs)  
- Chapter 8: Deep Learning Applications (image recognition, MNIST, etc.)

### Book 2: Natural Language Processing
- Chapter 1: Review of Neural Networks  
- Chapter 2: Natural Language and Distributed Representations  
- Chapter 3: Word2Vec Implementation  
- Chapter 4: Improving Word2Vec (negative sampling, subsampling)  
- Chapter 5: Recurrent Neural Networks (RNNs)  
- Chapter 6: Gated RNNs (LSTM / GRU)  
- Chapter 7: Sentence Generation with RNNs  
- Chapter 8: Attention Mechanisms  

### Book 3: DeZero and Advanced Deep Learning
- Chapter 1: Automatic Differentiation
- Chapter 2: Writing Clear & Pythonic Code
- Chapter 3: Higher-Order Differentiation
- Chapter 4: Building Neural Networks from Scratch
- Chapter 5: DeZero Challenges and Practical Features


*(Chapters may be updated as I progress through the book.)*

---

## 💡 What I Learned
- **From Book 1**:  
  - Implementing perceptrons and feedforward neural networks with NumPy  
  - Forward and backward propagation from scratch  
  - Gradient descent, backpropagation, and CNN basics  

- **From Book 2**:  
  - Building word embeddings and word2vec  
  - Implementing recurrent neural networks for sequence data  
  - Understanding the foundations of NLP models

- **From Book 3** *(in progress)*:  
  - Understanding reinforcement learning framework (agent, environment, reward, policy)  
  - Implementing bandit problems and policy evaluation methods  
  - Exploring Monte Carlo and temporal-difference methods  
  - Building simple DQN (Deep Q-Network) agents

---

## 🛠 How to Run
Clone the repository and run individual chapter scripts:

```bash
# Clone the repository
git clone https://github.com/markFT39/Deep-Learning-from-Scratch-series.git
cd "Deep-Learning-from-Scratch-series"

# Example (Book 1, Chapter 1)
cd "Deep Learning from Scratch (1)/Ch01"
python img_show.py

# Example (Book 2, Chapter 1)
cd "Deep Learning from Scratch (2)/Ch01"
python word2vec_basic.py

# Example (Book 3, Chapter 1)
cd "Deep Learning from Scratch (3)/steps"
python step01.py
