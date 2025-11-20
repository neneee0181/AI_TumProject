import numpy as np
import matplotlib.pyplot as plt
import torch

def predict_and_plot(model, X_test, y_test, scaler):
    model.eval()
    with torch.no_grad():
        preds = model(X_test).cpu().numpy()

    # 역정규화
    preds = scaler.inverse_transform(preds)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    plt.figure(figsize=(12,6))
    plt.plot(y_test, label="Real")
    plt.plot(preds, label="Predicted")
    plt.legend()
    plt.show()
