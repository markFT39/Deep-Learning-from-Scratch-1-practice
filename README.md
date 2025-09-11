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

- **Deep Learning from Scratch (3)** *(planned)*  
  Focus: Reinforcement Learning (RL) basics and implementations.

- **Deep Learning from Scratch (4)** *(planned)*  
  Focus: Generative Models (GANs, VAEs) and advanced applications.

---

## 📁 Contents

### Book 1: Fundamentals
- Chapter 1: Python and NumPy Basics  
- Chapter 2: Perceptron  
- Chapter 3: Neural Networks  
- Chapter 4: Learning Algorithms  
- Chapter 5: Backpropagation  
- Chapter 6: Neural Network Training Techniques  
- Chapter 7: CNN (Convolutional Neural Networks)  
- Chapter 8: Practical Examples  

### Book 2: Natural Language Processing
- Chapter 1: Word Vector Representations  
- Chapter 2: Word2Vec Implementation  
- Chapter 3: RNN and Language Modeling  
- Chapter 4: Gated RNNs (GRU, LSTM)  
- Chapter 5: seq2seq and Attention  
- Chapter 6: Practical NLP Tasks  

*(Chapters may be updated as I progress through the book.)*

---

## 💡 What I Learned
- **From Book 1**:  
  - Implementing perceptrons and feedforward neural networks with NumPy  
  - Forward and backward propagation from scratch  
  - Gradient descent, backpropagation, and CNN basics  

- **From Book 2** *(in progress)*:  
  - Building word embeddings and word2vec  
  - Implementing recurrent neural networks for sequence data  
  - Understanding the foundations of NLP models  

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
