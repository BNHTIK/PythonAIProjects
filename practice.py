import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
from PIL import Image
import os

class LetterPerceptron:
    def __init__(self, input_size, hidden_size, output_size=52, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.W1 = np.random.randn(self.input_size, self.hidden_size) / np.sqrt(self.input_size)
        self.b1 = np.zeros((1, self.hidden_size))

        self.W2 = np.random.randn(self.hidden_size, self.output_size) / np.sqrt(self.hidden_size)
        self.b2 = np.zeros((1, self.output_size))

        self.loss_history = []
        self.accuracy_history = []

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss

    def calculate_accuracy(self, y_true, y_pred):
        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
        accuracy = np.mean(y_true_labels == y_pred_labels)
        return accuracy

    def forward_propagation(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)

        return self.A2

    def backward_propagation(self, X, y_true, y_pred):
        m = X.shape[0]

        dZ2 = y_pred - y_true
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, X_train, y_train, X_val, y_val, epochs=1000, verbose=True):
        for epoch in range(epochs):
            y_pred_train = self.forward_propagation(X_train)

            loss = self.compute_loss(y_train, y_pred_train)
            self.loss_history.append(loss)

            self.backward_propagation(X_train, y_train, y_pred_train)

            if epoch % 50 == 0:
                val_pred = self.forward_propagation(X_val)
                val_accuracy = self.calculate_accuracy(y_val, val_pred)
                self.accuracy_history.append(val_accuracy)

                if verbose and epoch % 100 == 0:
                    print(f"Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {val_accuracy:.4f}")

    def predict(self, X):
        y_pred = self.forward_propagation(X)
        return np.argmax(y_pred, axis=1)

def load_letter_data():
    print("Data loading...")

    letters_df = pd.read_csv('english.csv')

    base_dir = os.path.dirname(os.path.abspath(__file__))

    images = []
    labels = []

    for _, row in letters_df.iterrows():
        img_path = os.path.join(base_dir, row['image'])
        image = Image.open(img_path).convert("L").resize((32,32))
        img_array = np.array(image).flatten()
        images.append(img_array)
        labels.append(row['label'])

    X = np.array(images)
    y = np.array(labels)

    print("Data loaded")
    print(letters_df.head(), letters_df.shape, X[0].shape, y.shape, len(np.unique(y)), sep="\n")

    return preprocess_data(X, y)

def one_hot_encode(data, num_classes):
    one_hot = np.zeros((data.shape[0], num_classes))
    one_hot[np.arange(data.shape[0]), data] = 1
    return one_hot

def preprocess_data(X, y):
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_train_onehot = one_hot_encode(y_train_encoded, num_classes=52)
    y_val_encoded = encoder.transform(y_val)
    y_val_onehot = one_hot_encode(y_val_encoded, num_classes=52)
    y_test_encoded = encoder.transform(y_test)
    y_test_onehot = one_hot_encode(y_test_encoded, num_classes=52)

    return(
        X_train, X_val, X_test,
        y_train_onehot, y_val_onehot, y_test_onehot,
        y_test, scaler, encoder
    )

def visualize_letters(X, y, scaler, num_samples=52):
    fig, axs = plt.subplots(6, 9, figsize=(12, 6))
    fig.suptitle("Dataset Letters Example", fontsize=16)

    print(type(X))
    print(type(scaler))

    X_denorm = scaler.inverse_transform(X)

    for i in range(num_samples):
        row = i // 9
        col = i % 9

        image = X_denorm[i].reshape(32, 32)
        axs[row, col].imshow(image, cmap='grey')
        axs[row, col].set_title(f"Letter: {y[i]}")
        axs[row, col].axis("off")

    plt.show()

def show_predictions(model, X_test, y_test_labels, scaler, encoder):
    print("Showing model predictions...")
    predictions = model.predict(X_test)
    predictions = encoder.inverse_transform(predictions)
    X_test_denorm = scaler.inverse_transform(X_test)

    indices = np.random.choice(len(X_test), size=52, replace=False)

    fig, axs = plt.subplots(6, 9, figsize=(18, 8))
    fig.suptitle("Model predictions", fontsize=16)

    for i, idx in enumerate(indices):
        row = i // 9
        col = i % 9

        image = X_test_denorm[idx].reshape(32, 32)
        axs[row, col].imshow(image, cmap='grey')

        correct = predictions[idx] == y_test_labels[idx]
        color = 'green' if correct else 'red'

        axs[row, col].set_title(f"True: {y_test_labels[idx]}\nPredicted: {predictions[idx]}", color=color)
        axs[row, col].axis('off')

    plt.show()

def main():
    X_train, X_val, X_test, y_train_onehot, y_val_onehot, y_test_onehot, y_test_labels, scaler, encoder = load_letter_data()
    visualize_letters(X_train, y_test_labels, scaler)

    input_size = X_train.shape[1]
    hidden_size = 100
    output_size = 52
    learning_rate = 0.01

    model = LetterPerceptron(input_size, hidden_size, output_size, learning_rate)

    epochs = 10000
    model.train(X_train, y_train_onehot, X_val, y_val_onehot, epochs, verbose=True)

    show_predictions(model, X_test, y_test_labels, scaler, encoder)

    return model

if __name__ == "__main__":
    model = main()