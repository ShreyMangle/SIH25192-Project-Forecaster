import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

df = pd.read_csv("data/synthetic_power_sector_projects.csv")

X = df.drop(columns=["Cost_Overrun_Pct", "Time_Overrun_Months"])
y = df[["Cost_Overrun_Pct", "Time_Overrun_Months"]]

categorical = X.select_dtypes(include="object").columns
numeric = X.select_dtypes(exclude="object").columns

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("cat", categorical_transformer, categorical),
    ("num", numeric_transformer, numeric)
])

rf = RandomForestRegressor(n_estimators=200, random_state=42)
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", MultiOutputRegressor(rf))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

joblib.dump(model, "models/multioutput_rf_pipeline.joblib")
print(" Model saved to models/multioutput_rf_pipeline.joblib")
