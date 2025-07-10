from multidimentional_regression import linear_regression, predict
import pandas as pd
import numpy as np
data = pd.read_csv("Housing.csv")
labels = data.price.to_numpy()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = pd.factorize(data[col])[0]

features = data.loc[:,"area":"furnishingstatus"].to_numpy()

weights = linear_regression(features, labels, 0.00001, 1000000)
comparison = pd.DataFrame({"Expected":labels, "Predicted":predict(weights, features).astype(int)})
comparison["Miss"] = abs(comparison["Expected"] - comparison["Predicted"])
print(comparison)
print(f"Average miss: {int(comparison["Miss"].mean())}")