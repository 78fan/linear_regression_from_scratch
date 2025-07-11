from multidimentional_regression import linear_regression, predict
import pandas as pd
import numpy as np
data = pd.read_csv("Housing.csv")
labels = data.price.to_numpy()
def one_hot_encode(df, column):
    categories = df[column].unique()
    for cat in categories[1:]:
        df[f"{column}_{cat}"] = (df[column] == cat).astype(int)
    df.drop(column, axis=1, inplace=True)
    return df


for col in data.select_dtypes(exclude=['number']).columns:
    data = one_hot_encode(data, col)

for column in data.columns:
    for i in [2, 3]:
        data[f'{column}{i}'] = data[column]**i


features = data.loc[:, data.columns != 'price'].to_numpy()

weights = linear_regression(features, labels, 0.1, 100000, 0.0001, 1)
print(data.columns)
print(weights)
comparison = pd.DataFrame({"Expected":labels, "Predicted":predict(weights, features).astype(int)})
comparison["Miss"] = abs(comparison["Expected"] - comparison["Predicted"])
print(comparison)
print(f"Average miss: {int(comparison["Miss"].mean())}")