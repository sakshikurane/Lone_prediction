import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and map form inputs to match model expectations
        gender = 1 if request.form.get('Gender') == 'Male' else 0
        married = 1 if request.form.get('Married') == 'Yes' else 0
        dependents = float(request.form.get('Dependents', 0))
        education = 1 if request.form.get('Education') == 'Graduate' else 0
        self_employed = 1 if request.form.get('Self_Employed') == 'Yes' else 0
        applicant_income = float(request.form.get('ApplicantIncome', 0))
        coapplicant_income = float(request.form.get('CoapplicantIncome', 0))
        loan_amount = float(request.form.get('LoanAmount', 0))
        loan_amount_term = float(request.form.get('Loan_Amount_Term', 360))
        credit_history = float(request.form.get('Credit_History', 1))
        
        prop_area = request.form.get('Property_Area', 'Urban')
        property_area_mapping = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
        property_area = property_area_mapping.get(prop_area, 2)

        # Assemble features into array
        features = np.array([[
            gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income, loan_amount,
            loan_amount_term, credit_history, property_area
        ]])

        prediction = model.predict(features)
        result = 'Loan Approved' if prediction[0] == 1 else 'Loan Rejected'

        return render_template('index.html', prediction_text=f'Result: {result}')
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
