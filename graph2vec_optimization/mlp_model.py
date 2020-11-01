import torch.nn as nn

class mlp_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 1. layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, int(hidden_size))  # 2.layer
        self.fc3 = nn.Linear(int(hidden_size), output_size)  # 3.layer
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        output = self.softmax(output)
        # print(output)
        # print(output.size())

        return output
