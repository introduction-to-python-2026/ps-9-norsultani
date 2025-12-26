import pandas as pd 
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("parkinsons.csv")
df = df.dropna()

df.head()

selected_features = df[["MDVP:Jitter(%)", "MDVP:Flo(Hz)"]]

x = selected_features
y = df["MDVP:RAP"]

x

scaler = MinMaxScaler()
x_scales = scaler.fit_transform(x)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scales, y, test_size = 0.2, )

len(x_train), len(x_test), len(y_train), len(y_test)

from sklearn.neighbors import KNeighborsRegressor


model = KNeighborsRegressor(n_neighbors = 3)


model.fit(x_train, y_train)

yd_pred = model.predict(x_test)

yd_pred


from sklearn.metrics import r2_score

r_squared = r2_score(y_test, yd_pred)

print(f"R-squared: {r_squared:.4f}")


import joblib

joblib.dump(model, 'parkinsons_predict.joblib')
