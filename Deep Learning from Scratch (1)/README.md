# Deep Learning from Scratch (1) - Practice

This directory contains my code implementations and notes based on **"밑바닥부터 시작하는 딥러닝 (1)"** by 사이토 고키 (Goki Saito).  
The focus of this book is to build **fundamental deep learning models from scratch** using only NumPy, without relying on frameworks such as TensorFlow or PyTorch.

📎 Related Link: [Repository](https://github.com/markFT39/Deep-Learning-from-Scratch-series)

---

## 📁 Contents
- **Chapter 1**: Python and NumPy Basics  
- **Chapter 2**: Perceptron  
- **Chapter 3**: Neural Networks  
- **Chapter 4**: Neural Network Training (loss functions, numerical differentiation, etc.)  
- **Chapter 5**: Backpropagation  
- **Chapter 6**: Training Techniques (weight initialization, overfitting prevention, hyperparameters)  
- **Chapter 7**: Convolutional Neural Networks (CNNs)  
- **Chapter 8**: Deep Learning Applications (image recognition, MNIST, etc.)

---

## 💡 What I Learned
- Implementing **basic perceptrons and logic gates** with NumPy  
- Constructing **feedforward neural networks** step by step  
- Understanding the math behind **forward propagation and backpropagation**  
- Implementing **gradient descent and numerical differentiation** for optimization  
- Training deep networks on datasets (e.g., **MNIST**) without frameworks  
- Exploring **convolutional neural networks (CNNs)** for image recognition  
- Applying practical **training techniques** such as mini-batch learning and weight initialization  

---

## 🛠 How to Run
Clone the repo and run chapter scripts inside the Book 1 folder:

```bash
git clone https://github.com/markFT39/Deep-Learning-from-Scratch-series.git
cd "Deep Learning from Scratch (1)"

# Example: Chapter 1
cd Ch01
python img_show.py

# Example: Chapter 3
cd ../Ch03
python neuralnet_mnist.py
