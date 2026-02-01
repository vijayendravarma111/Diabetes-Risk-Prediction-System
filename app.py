from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("diabetes_model.pkl")

FEATURE_COLUMNS = [
    'HighBP','HighChol','CholCheck','BMI','Smoker','Stroke',
    'HeartDiseaseorAttack','PhysActivity','Fruits','Veggies',
    'HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','GenHlth',
    'MentHlth','PhysHlth','DiffWalk','Sex','Age','Education','Income'
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    input_df = pd.DataFrame([[data[col] for col in FEATURE_COLUMNS]],
                            columns=FEATURE_COLUMNS)

    prob = model.predict_proba(input_df)[0][1]

    return jsonify({
        "probability": round(float(prob), 2),
        "risk": "High Risk of Diabetes" if prob >= 0.30 else "Low Risk of Diabetes"
    })

if __name__ == "__main__":
    app.run(debug=True)
