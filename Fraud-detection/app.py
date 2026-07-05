import os
import pickle
import warnings
from flask import Flask, render_template, request
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

model = pickle.load(open("Fraud-detection/model/model.pkl", "rb"))
scaler = pickle.load(open("Fraud-detection/model/scaler.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            input_features = [
                float(request.form["transaction_amount"]),
                float(request.form["hour_of_day"]),
                float(request.form["is_weekend"]),
                float(request.form["num_items"]),
                float(request.form["customer_age"]),
                float(request.form["prev_transactions"]),
                float(request.form["distance_from_home"]),
                float(request.form["device_type"]),
                float(request.form["network_quality"]),
                float(request.form["is_first_transaction"]),
                float(request.form["store_type"]),
                float(request.form["velocity_score"]),
            ]

            input_array = np.array(input_features).reshape(1, -1)
            scaled_features = scaler.transform(input_array)

            prediction = model.predict(scaled_features)

            cluster_id = int(prediction.item())

            if cluster_id == 0:
                user_friendly_result = "High Risk / Fraud Suspected"
                alert_color = "#dc3545"  # Red
                description = "Warning: This transaction matches parameters belonging to the highest historical fraud density segment (11.52%)."
            elif cluster_id == 1:
                user_friendly_result = "Medium Risk / Review Required"
                alert_color = "#ffc107"  # Yellow
                description = "Caution: This transaction shows intermediate patterns matching structural variance controls (10.52%)."
            else:
                user_friendly_result = "Low Risk / Standard Pattern"
                alert_color = "#28a745"  # Green
                description = "Safe: This transaction falls into the baseline normal consumer behavior profile with lowest risk metrics."

            return render_template(
                "result.html",
                result=user_friendly_result,
                color=alert_color,
                desc=description,
            )

        except Exception as e:
            return f"An operational error occurred during pipeline forwarding: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)
