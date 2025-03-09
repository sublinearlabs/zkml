import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


def normalize_value(value: int, mean: int, std_deviation: int):
    return (value - mean) / std_deviation

def denormalize_value(normalized_value: int, mean: int, std_deviation: int):
    return (normalized_value * std_deviation ) + mean 



x = np.array([x for x in range(100)])

reshaped_x = x.reshape(-1,1)


y = 46 + 2 * x.flatten()


plt.scatter(x, y, label="Initial data")
plt.title("Test data")
plt.xlabel("Values")
plt.ylabel("Time")

# plt.show()

x_mean, x_std = x.mean(), x.std()
y_mean, y_std = y.mean(), y.std()



x_tensor = torch.tensor((reshaped_x - x_mean) / x_std, dtype=torch.float32)
y_tensor = torch.tensor((y - y_mean) / y_std, dtype=torch.float32)



class LinearRegressionModel(nn.Module):
    def __init__(self, feature_in, feature_out):
        super().__init__()
        self.linear = nn.Linear(in_features=feature_in, out_features=feature_out)

    def forward(self, x):
        return self.linear(x).squeeze(1)


in_features = 1
out_features = 1

model = LinearRegressionModel(in_features, out_features)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

no_of_epochs = 20


for epoch in range(no_of_epochs):

    output = model(x_tensor)

    loss = criterion(output, y_tensor)

    optimizer.zero_grad()
    
    loss.backward()

    optimizer.step()

    print(f"Epoch: {epoch}/{no_of_epochs}, Loss: {loss}")


# model.eval()

input_tensor = torch.tensor((90 - x_mean) / x_std, dtype=torch.float32).view(1, -1)

print(input_tensor)

with torch.no_grad():
    normalized_prediction = model(input_tensor)

res = denormalize_value(normalized_prediction, y_mean, y_std)

print(res)

val = torch.randn(1, 1, dtype=torch.float32)

onnx_program = torch.onnx.dynamo_export(model, val)

onnx_program.save("linear_regression.onnx")