import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds for reproducibility
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

torch.set_default_dtype(torch.float32)

california = datasets.fetch_california_housing()
X = california.data
y = california.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)
    

def loss_fn(output, target, model, alpha):
    mse = (1 / (2 * output.size()[0])) * torch.sum((output - target)**2)
    l2_reg = torch.sum(model.linear.weight**2)
    return mse + alpha * l2_reg


alphas = np.logspace(-4, 4, 10)


def evaluate_alpha(alpha):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_values = []
    for train_index, val_index in kf.split(X_train):
        X_train_cv, X_val = X_train[train_index], X_train[val_index]
        y_train_cv, y_val = y_train[train_index], y_train[val_index]
        
        # Standardize features for this fold
        scaler = StandardScaler()
        X_train_cv_scaled = scaler.fit_transform(X_train_cv)
        X_val_scaled = scaler.transform(X_val)
        
        # Convert to PyTorch tensors
        X_train_cv_scaled = torch.as_tensor(X_train_cv_scaled, dtype=torch.float32)
        y_train_cv = torch.as_tensor(y_train_cv, dtype=torch.float32).view(-1, 1)
        X_val_scaled = torch.as_tensor(X_val_scaled, dtype=torch.float32)
        y_val = torch.as_tensor(y_val, dtype=torch.float32).view(-1, 1)

        # Initialize model and optimizer
        model = LinearRegression(X_train_cv_scaled.size()[1])
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Train the model
        num_epoch = 100
        for epoch in range(num_epoch):
            optimizer.zero_grad()
            output = model(X_train_cv_scaled)
            loss = loss_fn(output, y_train_cv, model, alpha)
            loss.backward()
            optimizer.step()
        
        # Evaluate on validation set
        with torch.no_grad():
            y_pred = model(X_val_scaled)
            mse = nn.MSELoss()(y_pred, y_val).item()
            mse_values.append(mse)
    
    return np.mean(mse_values)




best_alpha = None
best_mse = float('inf')
for alpha in alphas:
    mse = evaluate_alpha(alpha)
    if mse < best_mse:
        best_mse = mse
        best_alpha = alpha
print("Best alpha:", best_alpha)





scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = torch.as_tensor(X_train_scaled, dtype=torch.float32)
y_train = torch.as_tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_scaled = torch.as_tensor(X_test_scaled, dtype=torch.float32)
y_test = torch.as_tensor(y_test, dtype=torch.float32).view(-1, 1)

model = LinearRegression(X_train_scaled.size()[1])
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epoch = 100
for epoch in range(num_epoch):
    optimizer.zero_grad()
    output = model(X_train_scaled)
    loss = loss_fn(output, y_train, model, best_alpha)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    y_pred = model(X_test_scaled)
    mse = nn.MSELoss()(y_pred, y_test).item()
print("Mean squared error:", mse)


input = torch.as_tensor(np.array([-1.1551, -0.2863, -0.5207, -0.1717, -0.0303,  0.0674,  0.1951,  0.2853]), dtype=torch.float32)

output = model(input)

print(input)
print(output)

onnx_program = torch.onnx.dynamo_export(model, input)

onnx_program.save("linear_regression2.onnx")