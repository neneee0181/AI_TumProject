import torch
import torch.nn as nn
from model.lstm import StockLSTM

def train_model(X_train, y_train, epochs=30, lr=0.001):

    model = StockLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        output = model(X_train)
        loss = criterion(output.squeeze(), y_train)

        loss.backward()
        optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    return model
