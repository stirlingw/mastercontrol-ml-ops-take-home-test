from typing import Dict
import pandas as pd
import pickle


def predict_tabular_handler(predict_tabular_request: Dict):
    file_name = predict_tabular_request["file_name"]
    final_model = None
    with open("./models/ml/pickled_tabular_model.pkl", "rb") as file:
        final_model = pickle.load(file)

    test_data = pd.read_csv(f"./data/{file_name}")
    y_pred = final_model.predict(test_data.drop(["label"], axis=1)).tolist()

    return {
        "predictions": y_pred,
    }

