from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model files
model = joblib.load("Farmer-crop-and-fertilizer/model/crop_label_model.pkl")
scaler = joblib.load("Farmer-crop-and-fertilizer/model/crop_label_scaler.pkl")
encoder = joblib.load("Farmer-crop-and-fertilizer/model/crop_label_encoder.pkl")

#for fertilizer
fert_model = joblib.load("Farmer-crop-and-fertilizer/model/fertilizer_model.pkl")
fert_scaler = joblib.load("Farmer-crop-and-fertilizer/model/fertilizer_scaler.pkl")
fert_encoder = joblib.load("Farmer-crop-and-fertilizer/model/fertilizer_encoder.pkl")

soil_encoder = joblib.load("Farmer-crop-and-fertilizer/model/soil_encoder.pkl")
crop_encoder = joblib.load("Farmer-crop-and-fertilizer/model/crop_encoder.pkl")

remark_dict = joblib.load("Farmer-crop-and-fertilizer/model/fertilizer_remark.pkl")

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/crop")
def crop():
    return render_template("crop.html")


@app.route("/fertilization")
def fertilization():
    return render_template(
        "fertilizer.html",
        crops=crop_encoder.classes_,
        soils=soil_encoder.classes_
    )



@app.route("/predict", methods=["POST"])
def predict():
    try:
        # form data
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])
       
        # input array
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # scaling
        scaled = scaler.transform(features)

        # prediction
        pred = model.predict(scaled)

        # decode label
        crop_name = encoder.inverse_transform(pred)[0]

        return render_template("result.html", crop=crop_name)

    except Exception as e:
        return f"Error: {str(e)}"
        

@app.route("/predict_fertilizer", methods=["POST"])
def predict_fertilizer():

    # 1. form se data lo
    temp = float(request.form["Temperature"])
    moisture = float(request.form["Moisture"])
    rainfall = float(request.form["Rainfall"])
    ph = float(request.form["PH"])
    N = float(request.form["Nitrogen"])
    P = float(request.form["Phosphorous"])
    K = float(request.form["Potassium"])
    carbon = float(request.form["Carbon"])

    crop = request.form["Crop"].strip()
    soil = request.form["Soil"].strip()

    # safety check
    if crop not in crop_encoder.classes_:
        return f"Invalid crop: {crop}"

    if soil not in soil_encoder.classes_:
        return f"Invalid soil: {soil}"

# encode
    crop = crop_encoder.transform([crop])[0]
    soil = soil_encoder.transform([soil])[0]

    # 3. input array
    features = np.array([[temp, moisture, rainfall, ph, N, P, K, carbon, soil, crop]])

    # 4. scaling
    scaled = fert_scaler.transform(features)

    # 5. prediction
    pred = fert_model.predict(scaled)

    fertilizer = fert_encoder.inverse_transform(pred)[0]

    # 6. remark
    remark = remark_dict.get(fertilizer, "No remark available")

    # 7. result page
    return render_template("result.html",
                           fertilizer=fertilizer,
                           remark=remark)



if __name__ == "__main__":
    app.run(debug=True)
