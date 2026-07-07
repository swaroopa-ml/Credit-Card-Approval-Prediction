from flask import Flask, render_template, request
import joblib
import numpy as np

# Create Flask App
app = Flask(__name__)

# Load the trained model
model = joblib.load("credit_card_approval_model.pkl")


# -------------------- HOME PAGE --------------------
@app.route('/')
def home():
    return render_template("home.html")


# -------------------- PREDICTION PAGE --------------------
@app.route('/predict_page')
def predict_page():
    return render_template("index.html")


# -------------------- PREDICTION --------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read all input values from the form
        features = [
            float(request.form['ID']),
            float(request.form['CODE_GENDER']),
            float(request.form['FLAG_OWN_CAR']),
            float(request.form['FLAG_OWN_REALTY']),
            float(request.form['CNT_CHILDREN']),
            float(request.form['AMT_INCOME_TOTAL']),
            float(request.form['NAME_INCOME_TYPE']),
            float(request.form['NAME_EDUCATION_TYPE']),
            float(request.form['NAME_FAMILY_STATUS']),
            float(request.form['NAME_HOUSING_TYPE']),
            float(request.form['DAYS_BIRTH']),
            float(request.form['DAYS_EMPLOYED']),
            float(request.form['FLAG_MOBIL']),
            float(request.form['FLAG_WORK_PHONE']),
            float(request.form['FLAG_PHONE']),
            float(request.form['FLAG_EMAIL']),
            float(request.form['OCCUPATION_TYPE']),
            float(request.form['CNT_FAM_MEMBERS'])
        ]

        # Convert to NumPy array
        features = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]

        if prediction == 1:
            result = "Credit Card Approved"
        else:
            result = "Credit Card Rejected"

        return render_template("result.html", prediction=result)

    except Exception as e:
        return render_template("result.html", prediction=f"Error: {str(e)}")


# -------------------- RUN APPLICATION --------------------
if __name__ == "__main__":
    app.run(debug=True)