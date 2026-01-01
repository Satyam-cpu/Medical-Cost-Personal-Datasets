from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("model/insurance_best_model.joblib")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        data = {
            "age": int(request.form["age"]),
            "sex": request.form["sex"],
            "bmi": float(request.form["bmi"]),
            "children": int(request.form["children"]),
            "smoker": request.form["smoker"],
            "region": request.form["region"]
        }

        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        prediction = round(prediction, 2)

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
