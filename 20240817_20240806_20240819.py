"""
Neural Network from Scratch for Image Classification
Team: 20240806_20240817_20240819.py
Authors: Shahd Mostafa, Nourhan Nour, Ziad Mohamed

The network can be configured by the user (hidden layers, neurons, activation function).
It trains on the DIGITS dataset (scaled to MNIST format) from scikit-learn.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ============================================
# 1. DATA LOADING AND PREPROCESSING SECTION
# ============================================

def load_digits_mnist_format():
    """
    Load the DIGITS dataset from scikit-learn and convert it to MNIST-like format.
    
    DIGITS dataset characteristics:
    - 1797 total samples
    - 10 classes (digits 0-9)
    - Original size: 8x8 pixels (64 features)
    
    We convert it to 28x28 pixels (784 features) to match MNIST format.
    
    Steps:
    1. Load DIGITS dataset from scikit-learn
    2. Convert 8x8 images to 28x28 by adding padding
    3. Scale pixel values to 0-255 range (like MNIST)
    4. Split into train and test sets
    
    Returns:
        (train_images, train_labels), (test_images, test_labels)
    """
    print("\n📥 Loading DIGITS dataset from scikit-learn...")
    
    # Load the DIGITS dataset
    digits_data = load_digits()
    X_original = digits_data.data        # Shape: (1797, 64) - 8x8 flattened
    y_original = digits_data.target      # Shape: (1797,) - labels 0-9
    
    print(f"  • Original dataset size: {X_original.shape[0]} samples")
    print(f"  • Original image size: 8x8 pixels ({X_original.shape[1]} features)")
    print(f"  • Number of classes: {len(np.unique(y_original))} (digits 0-9)")
    
    # =========================================================================
    # CONVERT 8x8 IMAGES TO 28x28 (MNIST FORMAT)
    # =========================================================================
    print("\n🔄 Converting images from 8x8 to 28x28 (MNIST format)...")
    
    num_samples = X_original.shape[0]
    new_size = 28 * 28  # MNIST image size
    
    # Initialize array for resized images
    X_resized = np.zeros((num_samples, new_size))
    
    for i in range(num_samples):
        # Reshape to 8x8
        img_8x8 = X_original[i].reshape(8, 8)
        
        # Create 28x28 canvas filled with zeros (black background)
        img_28x28 = np.zeros((28, 28))
        
        # Calculate padding to center the 8x8 image
        pad_top = (28 - 8) // 2    # 10 pixels padding on top
        pad_left = (28 - 8) // 2   # 10 pixels padding on left
        
        # Place the 8x8 image in the center of the 28x28 canvas
        img_28x28[pad_top:pad_top+8, pad_left:pad_left+8] = img_8x8
        
        # Flatten back to 1D array
        X_resized[i] = img_28x28.flatten()
    
    # =========================================================================
    # SCALE PIXEL VALUES TO 0-255 RANGE (LIKE MNIST)
    # =========================================================================
    print("📊 Scaling pixel values to 0-255 range...")
    
    # Original DIGITS values are 0-16
    # Convert to 0-255 range like MNIST
    X_scaled = (X_resized / 16.0 * 255).astype(np.uint8)
    
    print(f"  • New image size: 28x28 pixels ({new_size} features)")
    print(f"  • Pixel range: {X_scaled.min()} to {X_scaled.max()}")
    
    # =========================================================================
    # SPLIT INTO TRAIN AND TEST SETS
    # =========================================================================
    print("\n✂️  Splitting data into train and test sets...")
    
    # Use 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_original, test_size=0.2, random_state=42, stratify=y_original
    )
    
    print(f"  • Training samples: {X_train.shape[0]} (80%)")
    print(f"  • Test samples: {X_test.shape[0]} (20%)")
    
    return (X_train, y_train), (X_test, y_test)


def preprocess_data(images, labels, num_classes=10):
    """
    Preprocess the images and labels for neural network training.
    
    Steps:
    1. Normalize pixel values from [0, 255] to [0, 1] for better training stability
    2. Convert labels to one-hot encoding
    
    Args:
        images: Image data (pixel values 0-255)
        labels: Labels (digits 0-9)
        num_classes: Number of classes (10 for digits 0-9)
    
    Returns:
        images_normalized: Images normalized to [0, 1]
        labels_onehot: One-hot encoded labels
        labels_original: Original labels (for reference)
    """
    # Normalize images to [0, 1] range
    images_normalized = images.astype(np.float32) / 255.0
    
    # One-hot encode labels
    labels_onehot = np.zeros((labels.shape[0], num_classes))
    labels_onehot[np.arange(labels.shape[0]), labels] = 1
    
    return images_normalized, labels_onehot, labels


def train_val_test_split(X, y, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Split data into training, validation, and test sets.
    
    Args:
        X: Input features (images)
        y: Labels (one-hot encoded)
        val_ratio: Percentage of data for validation
        test_ratio: Percentage of data for testing
        random_seed: For reproducibility
    
    Returns:
        (X_train, y_train): Training data
        (X_val, y_val): Validation data
        (X_test, y_test): Test data
    """
    np.random.seed(random_seed)
    
    # First split: separate test set
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_ratio)
    test_indices = indices[:test_size]
    remaining_indices = indices[test_size:]
    
    # Second split: separate validation set from remaining data
    val_size = int(len(remaining_indices) * val_ratio / (1 - test_ratio))
    val_indices = remaining_indices[:val_size]
    train_indices = remaining_indices[val_size:]
    
    # Create splits
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    print(f"  • Training set: {len(X_train)} samples")
    print(f"  • Validation set: {len(X_val)} samples")
    print(f"  • Test set: {len(X_test)} samples")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ============================================
# 2. ACTIVATION FUNCTIONS SECTION
# ============================================

class Activation:
    """Activation functions and their derivatives"""
    
    @staticmethod
    def sigmoid(x):
        """Sigmoid: f(x) = 1 / (1 + e^(-x))"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x):
        """Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))"""
        sig = Activation.sigmoid(x)
        return sig * (1 - sig)
    
    @staticmethod
    def relu(x):
        """ReLU: f(x) = max(0, x)"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        """Derivative of ReLU: f'(x) = 1 if x > 0, else 0"""
        return (x > 0).astype(np.float32)
    
    @staticmethod
    def softmax(x):
        """Softmax: converts logits to probabilities"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def get_activation(name):
        """Get activation function and its derivative by name"""
        activations = {
            'sigmoid': (Activation.sigmoid, Activation.sigmoid_derivative),
            'relu': (Activation.relu, Activation.relu_derivative)
        }
        return activations.get(name, (Activation.sigmoid, Activation.sigmoid_derivative))


# ============================================
# 3. LOSS FUNCTION SECTION
# ============================================

class Loss:
    """Loss functions for training"""
    
    @staticmethod
    def cross_entropy(y_pred, y_true, epsilon=1e-12):
        """
        Cross-entropy loss for multi-class classification
        
        Formula: L = -Σ y_true * log(y_pred)
        """
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    
    @staticmethod
    def cross_entropy_derivative(y_pred, y_true):
        """
        Derivative of cross-entropy with softmax
        
        When using softmax + cross-entropy:
        dL/dz = y_pred - y_true
        """
        return (y_pred - y_true) / y_true.shape[0]


# ============================================
# 4. NEURAL NETWORK IMPLEMENTATION
# ============================================

class NeuralNetwork:
    """Neural network implementation from scratch"""
    
    def __init__(self, layer_sizes, activation_func='sigmoid', learning_rate=0.01, random_seed=42):
        """
        Initialize neural network
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, ..., output]
            activation_func: 'sigmoid' or 'relu'
            learning_rate: Step size for gradient descent
            random_seed: For reproducible weight initialization
        """
        self.layer_sizes = layer_sizes
        self.activation_func = activation_func
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)
        
        # Get activation function
        self.activation, self.activation_derivative = Activation.get_activation(activation_func)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        np.random.seed(random_seed)
        
        for i in range(self.num_layers - 1):
            # Weight initialization strategy
            if activation_func == 'relu':
                scale = np.sqrt(2.0 / layer_sizes[i])  # He initialization
            else:
                scale = np.sqrt(1.0 / layer_sizes[i])  # Xavier initialization
            
            # Initialize weights with random values
            weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
            bias = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(weight)
            self.biases.append(bias)
        
        # Storage for forward pass values
        self.activations = []
        self.z_values = []
    
    def forward(self, X):
        """Forward propagation through the network"""
        self.activations = [X]
        self.z_values = []
        
        current_activation = X
        
        # Hidden layers
        for i in range(self.num_layers - 2):
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            current_activation = self.activation(z)
            self.activations.append(current_activation)
        
        # Output layer (softmax)
        z_output = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_output)
        output = Activation.softmax(z_output)
        self.activations.append(output)
        
        return output
    
    def backward(self, y_true):
        """Backward propagation to compute gradients"""
        m = y_true.shape[0]  # Batch size
        deltas = [None] * (self.num_layers - 1)
        
        # Output layer delta
        deltas[-1] = Loss.cross_entropy_derivative(self.activations[-1], y_true)
        
        # Backpropagate through hidden layers
        for l in range(self.num_layers - 2, 0, -1):
            delta = np.dot(deltas[l], self.weights[l].T) * self.activation_derivative(self.z_values[l-1])
            deltas[l-1] = delta
        
        # Update weights and biases
        for l in range(self.num_layers - 1):
            grad_w = np.dot(self.activations[l].T, deltas[l]) / m
            grad_b = np.sum(deltas[l], axis=0, keepdims=True) / m
            
            self.weights[l] -= self.learning_rate * grad_w
            self.biases[l] -= self.learning_rate * grad_b
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, verbose=True):
        """
        Train the neural network
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of training epochs
            batch_size: Mini-batch size
            verbose: Whether to print progress
        
        Returns:
            Training history (losses and accuracies)
        """
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        print(f"\nStarting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Mini-batch gradient descent
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            for i in range(0, len(X_train), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward and backward pass
                output = self.forward(X_batch)
                self.backward(y_batch)
            
            # Calculate metrics
            train_pred = self.forward(X_train)
            train_loss = Loss.cross_entropy(train_pred, y_train)
            train_acc = self.accuracy(X_train, np.argmax(y_train, axis=1))
            
            val_pred = self.forward(X_val)
            val_loss = Loss.cross_entropy(val_pred, y_val)
            val_acc = self.accuracy(X_val, np.argmax(y_val, axis=1))
            
            # Store history
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%} | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")
        
        print(f"\nTraining completed! Final validation accuracy: {val_acc:.2%}")
        return train_losses, val_losses, train_accuracies, val_accuracies
    
    def predict(self, X):
        """Make predictions for input samples"""
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def accuracy(self, X, y_true):
        """Calculate classification accuracy"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y_true)


# ============================================
# 5. MAIN PROGRAM - USER INTERFACE
# ============================================

def main():
    print("=" * 70)
    print("NEURAL NETWORK for Digit CLASSIFICATION")
    print("Dataset: DIGITS (scaled to MNIST format) from scikit-learn")
    print("=" * 70)
    
    # =========================================================================
    # STEP 1: LOAD AND PREPARE DATA
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: LOADING AND PREPROCESSING DATASET")
    print("="*70)
    
    # Load DIGITS dataset in MNIST format
    (train_images, train_labels), (test_images, test_labels) = load_digits_mnist_format()
    
    print("\n🔧 Preprocessing data...")
    X_full, y_full_onehot, y_full = preprocess_data(train_images, train_labels)
    
    print("\n✂️  Splitting data into train/validation/test sets...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(
        X_full, y_full_onehot, val_ratio=0.15, test_ratio=0.15
    )
    
    # =========================================================================
    # STEP 2: GET USER INPUT FOR NETWORK CONFIGURATION
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: CONFIGURE YOUR NEURAL NETWORK")
    print("="*70)
    
    print("\n📊 Dataset Specifications:")
    print(f"  • Input size: 784 (28x28 pixel images)")
    print(f"  • Output size: 10 (digits 0-9)")
    print(f"  • Training samples: {len(X_train)}")
    print(f"  • Validation samples: {len(X_val)}")
    print(f"  • Test samples: {len(X_test)}")
    
    # Network architecture
    input_size = 784
    output_size = 10
    
    # Get number of hidden layers
    print("\n" + "-"*50)
    print("HIDDEN LAYERS CONFIGURATION")
    print("-"*50)
    
    while True:
        try:
            num_hidden_layers = int(input("\nEnter number of hidden layers: "))
            if num_hidden_layers > 0:
                print(f"✓ Selected: {num_hidden_layers} hidden layer(s)")
                break
            else:
                print("⚠️  Please enter a number greater than 0")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
    
    # Get neurons in each hidden layer
    layer_sizes = [input_size]
    
    print("\n" + "-"*50)
    print("NEURONS PER HIDDEN LAYER")
    print("-"*50)
    
    for i in range(num_hidden_layers):
        while True:
            try:
                neurons = int(input(f"\nEnter neurons in hidden layer {i+1}: "))
                if 1 <= neurons <= 512: # Limit to reasonable size for this dataset to avoid overfitting
                    layer_sizes.append(neurons)
                    print(f"✓ Layer {i+1}: {neurons} neurons")
                    break
                else:
                    print("⚠️  Please enter a number between 1 and 512")
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
    
    layer_sizes.append(output_size)
    
    # Get activation function
    print("\n" + "-"*50)
    print("ACTIVATION FUNCTION")
    print("-"*50)
    
    while True:
        activation = input("\nChoose activation (sigmoid/relu): ").lower()
        if activation in ['sigmoid', 'relu']:
            print(f"✓ Selected: {activation}")
            break
        else:
            print("❌ Please choose either 'sigmoid' or 'relu'")
    
    # Get learning rate
    print("\n" + "-"*50)
    print("LEARNING RATE")
    print("-"*50)
    
    while True:
        try:
            learning_rate = float(input("\nEnter learning rate: "))
            if learning_rate > 0:
                print(f"✓ Learning rate: {learning_rate}")
                break
            else:
                print("⚠️  Please enter a value greater than 0")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
    
    # =========================================================================
    # STEP 3: CREATE AND TRAIN THE NETWORK
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: CREATING AND TRAINING NEURAL NETWORK")
    print("="*70)
    
    print(f"\n🧠 Network Architecture: {layer_sizes}")
    print(f"   Activation: {activation}")
    print(f"   Learning rate: {learning_rate}")
    
    # Create neural network
    nn = NeuralNetwork(
        layer_sizes=layer_sizes,
        activation_func=activation,
        learning_rate=learning_rate,
        random_seed=42
    )
    
    # Train the network
    print("\n🚀 Starting training...")
    train_losses, val_losses, train_accs, val_accs = nn.train(
        X_train, y_train, X_val, y_val,
        epochs=100,        # More epochs for smaller dataset
        batch_size=32,
        verbose=True
    )
    
    # =========================================================================
    # STEP 4: TEST THE NETWORK
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: EVALUATING ON TEST SET")
    print("="*70)
    
    # Prepare test data
    test_images_normalized = test_images.astype(np.float32) / 255.0
    
    # Calculate accuracy
    test_acc = nn.accuracy(test_images_normalized, test_labels)
    
    print(f"\n📊 Test Set Results:")
    print(f"  • Test accuracy: {test_acc:.2%}")
    print(f"  • Correct predictions: {int(test_acc * len(test_labels))}/{len(test_labels)}")

# ============================================
# PROGRAM ENTRY POINT
# ============================================

if __name__ == "__main__":
    """
    Entry point of the program.
    """
    main()
