import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("data/synthetic_power_sector_projects.csv")
X = df.drop(columns=["Cost_Overrun_Pct", "Time_Overrun_Months"])
y = df[["Cost_Overrun_Pct", "Time_Overrun_Months"]]

model = joblib.load("models/multioutput_rf_pipeline.joblib")

pred = model.predict(X)

print(" Model Evaluation")
print("MSE:", mean_squared_error(y, pred))
print("R2 Score:", r2_score(y, pred))
