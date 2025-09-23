# Deep Learning from Scratch (2) - Practice

This directory contains my code implementations and notes based on **"Deep Learning from Scratch 2"** by Goki Saito.  
The focus of this book is on **recurrent neural networks (RNNs) and natural language processing (NLP)**, and all implementations are done from scratch with NumPy and Python.

📎 Related Link: [Repository](https://github.com/markFT39/Deep-Learning-from-Scratch-series)

---

## 📁 Contents
- **Chapter 1**: Review of Neural Networks  
- **Chapter 2**: Natural Language and Distributed Representations  
- **Chapter 3**: Word2Vec Implementation 
- **Chapter 4**: Improving Word2Vec (negative sampling, subsampling) 
- **Chapter 5**: Recurrent Neural Networks (RNNs)  
- **Chapter 6**: Gated RNNs (LSTM / GRU)  
- **Chapter 7**: Sentence Generation with RNNs  
- **Chapter 8**: Attention Mechanisms  

---

## 💡 What I Learned (highlights)
- Recap of neural network fundamentals required for sequence models.  
- Word embedding concepts and practical implementation of **word2vec** (CBOW / Skip-gram).  
- Techniques to speed up embedding training (negative sampling, hierarchical softmax ideas covered).  
- Building and training **RNNs**, and why gated variants (LSTM/GRU) are necessary to mitigate vanishing gradients.  
- Using RNNs for generative tasks (simple text generation).  
- Introduction to **attention** and how it improves sequence modelling (foundation for seq2seq/transformer ideas).

---

## 🛠 How to Run (examples)
Clone the repo and run chapter scripts inside the Book 2 folder:

```bash
git clone https://github.com/markFT39/Deep-Learning-from-Scratch-series.git
cd "Deep Learning from Scratch (2)"

# Example: Chapter 3
cd ../Ch03
python train.py

# Example: Chapter 7
cd ../Ch07
python train_seq2seq.py