import joblib
from src.preprocess import prepare_input

model = joblib.load("models/multioutput_rf_pipeline.joblib")

user_input = {
    "State": "Maharashtra",
    "Project_Type": "Solar",
    "Planned_Cost_Cr": 500,
    "Actual_Cost_Cr": 650,
    "Duration_Months": 36,
    "Contractor_Experience_Level": "Medium",
    "Type_Terrain": "Hilly",
    "Start_Year": 2020,
    "Capacity_MW_or_km": 200,
    "Region": "West",
    "Budget_Category": "Medium",
    "Project_Size_Category": "Large",
    "Type_Experience": "National",
    "Planned_Duration_Months": 30,
    "Planned_Duration_per_Unit": 0.15,
    "Planned_Cost_per_Unit": 2.5,
    "Era": "Modern",
    "Terrain": "Hilly"
}

df = prepare_input(user_input, model)

prediction = model.predict(df)[0]
print("Predicted Cost Overrun %:", prediction[0])
print("Predicted Time Overrun (Months):", prediction[1])
