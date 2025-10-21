import torch
import torch.nn as nn

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


if __name__ == '__main__':
    model_full = torch.load('saved_models/model_full_20251021_150236.pth', weights_only=False)
    model_state = torch.load('saved_models/model_state_20251021_150236.pth', weights_only=True)

    model = SimpleNet()
    model.load_state_dict(model_state)
    model.eval()

    test_values = [28, 14, 75, 35, 12]

    for val in test_values:
        input_tensor = torch.FloatTensor([[val]])
        expected = val * 2

        predicted = model_full(input_tensor).item()
        print(f"(Full Model) Input: {val}, Predicted: {predicted}, Expected: {expected}")

        predicted = model(input_tensor).item()
        print(f"(State Model) Input: {val}, Predicted: {predicted}, Expected: {expected}")

