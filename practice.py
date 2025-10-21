import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
from datetime import datetime

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 3) # Прихованний шар з 3 нейронами і 1 входом
        self.output = nn.Linear(3, 1) # Вихідний шар з 1 нейроном і 3 входами
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x


model = SimpleNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

loss_history = []
epochs = 1000

for epoch in range(epochs):
    inputs = torch.tensor([[1], [5], [3], [2], [4]], dtype=torch.float32)
    targets = inputs * 2

    outputs = model.forward(inputs)
    loss = criterion(outputs, targets)
    loss_history.append(loss.item())

    print(f"Epoch #{epoch} loss: {loss.item():.6f}")

    optimizer.zero_grad() # Обнулення градієнтів
    loss.backward() # Обчислення градієнтів
    optimizer.step() # Оновлення ваг

# Збереження моделі
models_dir = 'saved_models'
os.makedirs(models_dir, exist_ok=True)

# Збереження усії моделі
model_path_full = os.path.join(models_dir, f"model_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
torch.save(model, model_path_full)

# Збереження лише параметрів моделі
model_path_state = os.path.join(models_dir, f"model_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
torch.save(model.state_dict(), model_path_state)

plt.plot(range(epochs), loss_history, color='red', label="Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train loss")
plt.grid(True)
plt.legend()
plt.show()

test_input = torch.tensor([[16]], dtype=torch.float32)
predicted = model(test_input)

print(f"Input: {test_input.item()}, Predicted: {predicted.item()}")
