import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load("models/multioutput_rf_pipeline.joblib")

model = load_model()

def prepare_input(user_input):
    expected_cols = [
        'Contractor_Experience_Level', 'Type_Terrain', 'Planned_Duration_per_Unit',
        'Era', 'Capacity_MW_or_km', 'Region', 'Start_Year',
        'Project_Size_Category', 'Planned_Cost_per_Unit',
        'Planned_Duration_Months', 'Budget_Category',
        'Type_Experience', 'Terrain', 'Project_Type', 'State',
        'Planned_Cost_Cr', 'Duration_Months'
    ]
    df = pd.DataFrame([user_input])
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    return df[expected_cols]

def main():
    st.title("⚡ Power Sector Project Forecaster")
    st.markdown("Predict **Cost Overrun** and **Time Overrun** using ML")

    st.subheader(" Enter Project Details")

    state = st.selectbox("State", ["Maharashtra", "Gujarat", "Karnataka", "Delhi", "Other"])
    project_type = st.selectbox("Project Type", ["Thermal", "Hydro", "Solar", "Wind", "Nuclear"])
    planned_cost = st.number_input("Planned Cost (₹ Cr)", min_value=100, max_value=10000, step=100)
    duration = st.number_input("Planned Duration (Months)", min_value=1, max_value=120, step=1)
    contractor = st.selectbox("Contractor Experience", ["Low", "Medium", "High"])
    terrain = st.selectbox("Terrain", ["Plain", "Hilly", "Coastal"])
    region = st.selectbox("Region", ["North", "South", "East", "West"])
    start_year = st.number_input("Start Year", min_value=2000, max_value=2030, step=1)

    if st.button(" Predict Overruns"):
        user_input = {
            "State": state,
            "Project_Type": project_type,
            "Planned_Cost_Cr": planned_cost,
            "Duration_Months": duration,
            "Contractor_Experience_Level": contractor,
            "Type_Terrain": terrain,
            "Start_Year": start_year,
            "Capacity_MW_or_km": 200,   
            "Region": region,
            "Budget_Category": "Medium",   
            "Project_Size_Category": "Large",  
            "Type_Experience": "National",  
            "Planned_Duration_Months": duration,
            "Planned_Duration_per_Unit": 0.15,  
            "Planned_Cost_per_Unit": 2.5,       
            "Era": "Modern",
            "Terrain": terrain
        }

        new_data = prepare_input(user_input)
        pred = model.predict(new_data)[0]

        st.success(f" Predicted Cost Overrun: **{pred[0]:.2f}%**")
        st.success(f" Predicted Time Overrun: **{pred[1]:.2f} months**")

if __name__ == "__main__":
    main()
