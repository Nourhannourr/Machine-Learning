"""
Neural Network from Scratch for Image Classification
Team: 20240806_20240817_20240819.py
Authors: Shahd Mostafa, Nourhan Nour, Ziad Mohamed
This code implements a customizable feedforward neural network to classify images from the MNIST dataset.

Features:
- Loading MNIST data from a local .npz file.
- User-defined architecture (number of hidden layers and neurons).
- Choice of activation functions (ReLU or Sigmoid).
- Training with backpropagation and gradient descent.
- Interactive prediction on test images with visualization.
- Modular structure for easy understanding and modification.
- Clear prompts and validation for user inputs.
- Comprehensive comments for clarity.
"""

import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# 1) LOAD MNIST FROM mnist.npz
# ================================================================
def load_mnist_npz():
    data = np.load("mnist.npz")

    X_train = data["x_train"]
    y_train = data["y_train"]
    X_test = data["x_test"]
    y_test = data["y_test"]

    # Normalize pixel values to [0,1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Flatten 28×28 → 784
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)

    # One-hot encode labels
    num_classes = 10
    y_train_oh = np.eye(num_classes)[y_train]
    y_test_oh = np.eye(num_classes)[y_test]

    return X_train, y_train_oh, X_test, y_test_oh, y_train, y_test


# ================================================================
# 2) ACTIVATION FUNCTIONS
# ================================================================
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s*(1-s)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)


# ================================================================
# 3) SOFTMAX + LOSS
# ================================================================
def softmax(z): 
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True) #keepdims to maintain shape as 2D array

def cross_entropy(pred, true):
    return -np.mean(np.sum(true * np.log(pred + 1e-8), axis=1))


# ================================================================
# 4) NEURAL NETWORK CLASS (UNCHANGED)
# ================================================================
class NeuralNetwork:
    def __init__(self, layer_sizes, activation="relu", learning_rate=0.01):
        self.learning_rate = learning_rate

        if activation == "relu":
            self.act = relu
            self.act_deriv = relu_deriv
        else:
            self.act = sigmoid
            self.act_deriv = sigmoid_deriv

        self.weights = []
        self.biases = []

        # Xavier Initialization for weights and zeros for biases
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        activations = [X]
        zs = []

        for i in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i] # z=wx+b
            zs.append(z) # store pre-activation values
            a = self.act(z) # apply activation function either ReLU or Sigmoid
            activations.append(a) # store activations

        z = activations[-1] @ self.weights[-1] + self.biases[-1]
        zs.append(z)
        probs = softmax(z) # output layer with softmax
        activations.append(probs) # store final output activations

        return activations, zs

    def backward(self, activations, zs, y_true):
        grads_w = []
        grads_b = []
        m = len(y_true)
        delta = activations[-1] - y_true

        for i in reversed(range(len(self.weights))):
            a_prev = activations[i]

            grad_w = (a_prev.T @ delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m

            grads_w.insert(0, grad_w)
            grads_b.insert(0, grad_b)

            if i != 0:
                delta = (delta @ self.weights[i].T) * self.act_deriv(zs[i-1])

        return grads_w, grads_b

    def update(self, grads_w, grads_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    def train(self, X, y, epochs=10, batch_size=128):
        n = len(X)
        for ep in range(epochs):
            idx = np.random.permutation(n)
            X, y = X[idx], y[idx]

            losses = []
            for i in range(0, n, batch_size):
                Xb = X[i:i+batch_size]
                yb = y[i:i+batch_size]

                activations, zs = self.forward(Xb)
                loss = cross_entropy(activations[-1], yb)
                losses.append(loss)

                grads_w, grads_b = self.backward(activations, zs, yb)
                self.update(grads_w, grads_b)

            print(f"Epoch {ep+1}/{epochs}   Loss: {np.mean(losses):.4f}")

    def predict(self, X):
        a, _ = self.forward(X)
        return np.argmax(a[-1], axis=1)


# ================================================================
# 5) USER INTERACTIVE SETUP + VALIDATION
# ================================================================
print("\n==============================")
print("  Fully Customizable MNIST NN")
print("==============================")

# ---- Hidden layers input ----
while True:
    try:
        num_hidden = int(input("How many hidden layers? : "))
        if num_hidden < 0:
            raise ValueError
        break
    except ValueError:
        print("❌ Please enter a valid non-negative integer.")

hidden_layers = []
for i in range(num_hidden):
    while True:
        try:
            nodes = int(input(f"Number of neurons for layer {i+1}: "))
            if nodes <= 0:
                raise ValueError
            hidden_layers.append(nodes)
            break
        except ValueError:
            print("❌ Please enter a positive integer for neurons.")

# ---- Activation function input ----
print("\nChoose activation: 1 = ReLU, 2 = Sigmoid")
while True:
    act_choice = input("Your choice: ")
    if act_choice in ["1", "2"]:
        activation = "relu" if act_choice == "1" else "sigmoid"
        break
    else:
        print("❌ Enter 1 or 2 only.")

# ---- Learning rate ----
while True:
    try:
        learning_rate = float(input("\nLearning rate : "))
        if learning_rate <= 0:
            raise ValueError
        break
    except ValueError:
        print("❌ Enter a positive number.")

# ---- Epochs ----
while True:
    try:
        epochs = int(input("Number of epochs : "))
        if epochs <= 0:
            raise ValueError
        break
    except ValueError:
        print("❌ Enter a positive integer.")


# ================================================================
# 6) LOAD DATA
# ================================================================
print("\nLoading mnist.npz ...")
X_train, y_train_oh, X_test, y_test_oh, y_train_raw, y_test_raw = load_mnist_npz()

# Train/Val split
val_split = 0.1
val_size = int(len(X_train) * val_split)

X_val = X_train[:val_size]
y_val = y_train_oh[:val_size]

X_train2 = X_train[val_size:]
y_train2 = y_train_oh[val_size:]

# ================================================================
# 7) BUILD + TRAIN NETWORK
# ================================================================
layers = [784] + hidden_layers + [10]
nn = NeuralNetwork(layers, activation=activation, learning_rate=learning_rate)

print("\nTraining Network...")
nn.train(X_train2, y_train2, epochs=epochs, batch_size=128)


# ================================================================
# 8) ACCURACY
# ================================================================
pred_test = nn.predict(X_test)
accuracy = np.mean(pred_test == y_test_raw)

print("\n==============================")
print(" Final Test Accuracy:", accuracy)
print("==============================")

# ================================================================
# 9) INTERACTIVE PREDICT + IMAGE DISPLAY
# ================================================================
while True:
    try:
        idx = int(input("\nEnter test image index (0–9999) or -1 to exit: "))
        if idx == -1:
            break
        if not (0 <= idx < len(X_test)):
            raise ValueError
    except ValueError:
        print("❌ Please enter a valid index between 0 and 9999.")
        continue

    img_flat = X_test[idx]
    img = img_flat.reshape(28, 28)

    pred = nn.predict(img_flat.reshape(1, -1))[0]

    print(f"True label: {y_test_raw[idx]}  |  Predicted: {pred}")

    plt.imshow(img, cmap="gray")
    plt.title(f"Prediction: {pred} | True: {y_test_raw[idx]}")
    plt.axis("off")
    plt.show()
