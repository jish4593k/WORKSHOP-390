import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Label, Button, filedialog

class PolynomialRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(PolynomialRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

def create_polynomial_dataset(N, D, noise):
    X, y = make_regression(n_samples=N, n_features=D, noise=noise)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return torch.Tensor(X_train), torch.Tensor(y_train), torch.Tensor(X_test), torch.Tensor(y_test)

def train_torch_model(model, dataset, lr=0.01, epochs=100):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(dataset[0])
        loss = criterion(outputs, dataset[1])
        loss.backward()
        optimizer.step()

    return model

def create_keras_model(input_size):
    model = Sequential()
    model.add(Dense(1, input_dim=input_size, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_keras_model(model, dataset, epochs=100, batch_size=32):
    model.fit(dataset[0], dataset[1], epochs=epochs, batch_size=batch_size, verbose=0)
    return model

def visualize_results(model, testset, results_savepath):
    test_outputs = model(testset[0])

    plt.scatter(testset[0].numpy(), testset[1].numpy(), color='blue', label='Actual')
    plt.scatter(testset[0].numpy(), test_outputs.detach().numpy(), color='red', label='Predicted')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.legend()
    plt.title('Polynomial Regression')
    plt.savefig(results_savepath)
    plt.show()

def main():
    N = 100
    D = 1
    noise = 10

    dataset = create_polynomial_dataset(N, D, noise)
    testset = dataset[2:4]  # Use the last 20% of the data for testing

    
    torch_model = train_torch_model(PolynomialRegressionModel(D), dataset)
    torch_test_outputs = torch_model(testset[0])
    torch_test_loss = mean_squared_error(torch_test_outputs.detach().numpy(), testset[1].numpy())
    print(f"PyTorch Test Loss: {torch_test_loss}")

    
    keras_model = train_keras_model(create_keras_model(D), dataset)
    keras_test_outputs = keras_model.predict(testset[0])
    keras_test_loss = mean_squared_error(keras_test_outputs, testset[1].numpy())
    print(f"Keras Test Loss: {keras_test_loss}")

  
    root = tk.Tk()
    root.title("Polynomial Regression Results")

    tk.Label(root, text="PyTorch Test Loss: {:.5f}".format(torch_test_loss)).pack()
    tk.Label(root, text="Keras Test Loss: {:.5f}".format(keras_test_loss)).pack()

    visualize_button = Button(root, text="Visualize Results", command=lambda: visualize_results(torch_model, testset, "results.png"))
    visualize_button.pack()

    root.mainloop()

if name_== "main":
    main()
