from flask import Flask, render_template,request
import numpy as np
import pickle
app=Flask (__name__)
with open("Female-diabetes-predictor/model/diabetes.pkl","rb") as file:
    model=pickle.load(file)

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict',methods=["POST"])
def predict():
    if request.method=="POST":
        try:
            pregnancies = float(request.form["Pregnancies"])
            glucose = float(request.form["Glucose"])
            bp = float(request.form["BloodPressure"])
            skin = float(request.form["SkinThickness"])
            insulin = float(request.form["Insulin"])
            bmi = float(request.form["BMI"])
            dpf = float(request.form["DiabetesPedigreeFunction"])
            age = float(request.form["Age"])

            input_features=[[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
            ]

            prediction=model.predict(input_features)

            if prediction[0]==1:
                result_text=" High Risk: You have high chances of being diabetic."
            else:
                result_text = "Safe: You have low or no chances of being diabetic."

            return render_template("result.html", prediction_text=result_text)

        except Exception as e:
            return render_template(
                "diab_index.html",prediction_text=f"Error :{str(e)}"
            )

if(__name__)=="__main__":
    app.run(debug=True)