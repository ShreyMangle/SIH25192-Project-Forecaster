import pandas as pd

def prepare_input(user_input: dict, model):

    expected_cols = model.feature_names_in_
    df = pd.DataFrame([user_input])
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    return df[expected_cols]
