import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained logistic regression model
with open('logistic.pkl', 'rb') as f:
    model = pickle.load(f)

# HTML Template with a Modern, Attractive UI Design
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loan Eligibility Prediction Portal</title>
    <!-- Bootstrap 5 CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/assets/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            padding: 40px 0;
        }
        .main-card {
            border: none;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            background: #ffffff;
        }
        .card-header {
            background: linear-gradient(135deg, #4e73df 0%, #224abe 100%);
            color: white;
            border-top-left-radius: 15px !important;
            border-top-right-radius: 15px !important;
            text-align: center;
            padding: 20px;
        }
        .form-label {
            font-weight: 600;
            color: #4a5568;
        }
        .btn-custom {
            background: linear-gradient(135deg, #4e73df 0%, #224abe 100%);
            border: none;
            color: white;
            font-weight: 600;
            padding: 12px;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        .btn-custom:hover {
            background: linear-gradient(135deg, #224abe 0%, #1a365d 100%);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        .result-box {
            border-radius: 10px;
            padding: 20px;
            margin-top: 25px;
            text-align: center;
            font-weight: bold;
            font-size: 1.2rem;
        }
        .approved {
            background-color: #d1e7dd;
            color: #0f5132;
            border: 1px solid #badbcc;
        }
        .rejected {
            background-color: #f8d7da;
            color: #842029;
            border: 1px solid #f5c2c7;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-lg-8">
                <div class="card main-card">
                    <div class="card-header">
                        <h2>Loan Eligibility Predictor</h2>
                        <p class="mb-0 text-white-50">Enter applicant details below to check loan approval status</p>
                    </div>
                    <div class="card-body p-4">
                        <form method="POST" action="/">
                            <div class="row g-3">
                                <!-- Gender -->
                                <div class="col-md-6">
                                    <label class="form-label">Gender</label>
                                    <select name="Gender" class="form-select" required>
                                        <option value="1">Male</option>
                                        <option value="0">Female</option>
                                    </select>
                                </div>
                                <!-- Married -->
                                <div class="col-md-6">
                                    <label class="form-label">Married</label>
                                    <select name="Married" class="form-select" required>
                                        <option value="1">Yes</option>
                                        <option value="0">No</option>
                                    </select>
                                </div>
                                <!-- Dependents -->
                                <div class="col-md-6">
                                    <label class="form-label">Dependents</label>
                                    <select name="Dependents" class="form-select" required>
                                        <option value="0">0</option>
                                        <option value="1">1</option>
                                        <option value="2">2</option>
                                        <option value="3">3+</option>
                                    </select>
                                </div>
                                <!-- Education -->
                                <div class="col-md-6">
                                    <label class="form-label">Education</label>
                                    <select name="Education" class="form-select" required>
                                        <option value="1">Graduate</option>
                                        <option value="0">Not Graduate</option>
                                    </select>
                                </div>
                                <!-- Self Employed -->
                                <div class="col-md-6">
                                    <label class="form-label">Self Employed</label>
                                    <select name="Self_Employed" class="form-select" required>
                                        <option value="0">No</option>
                                        <option value="1">Yes</option>
                                    </select>
                                </div>
                                <!-- Credit History -->
                                <div class="col-md-6">
                                    <label class="form-label">Credit History</label>
                                    <select name="Credit_History" class="form-select" required>
                                        <option value="1">Yes (1.0)</option>
                                        <option value="0">No (0.0)</option>
                                    </select>
                                </div>
                                <!-- Applicant Income -->
                                <div class="col-md-6">
                                    <label class="form-label">Applicant Income ($)</label>
                                    <input type="number" step="any" name="ApplicantIncome" class="form-control" placeholder="e.g. 5000" required>
                                </div>
                                <!-- Coapplicant Income -->
                                <div class="col-md-6">
                                    <label class="form-label">Coapplicant Income ($)</label>
                                    <input type="number" step="any" name="CoapplicantIncome" class="form-control" placeholder="e.g. 1500" required>
                                </div>
                                <!-- Loan Amount -->
                                <div class="col-md-6">
                                    <label class="form-label">Loan Amount ($ in thousands)</label>
                                    <input type="number" step="any" name="LoanAmount" class="form-control" placeholder="e.g. 120" required>
                                </div>
                                <!-- Loan Amount Term -->
                                <div class="col-md-6">
                                    <label class="form-label">Loan Amount Term (Months)</label>
                                    <input type="number" name="Loan_Amount_Term" class="form-control" value="360" required>
                                </div>
                                <!-- Property Area -->
                                <div class="col-12">
                                    <label class="form-label">Property Area</label>
                                    <select name="Property_Area" class="form-select" required>
                                        <option value="2">Urban</option>
                                        <option value="1">Semiurban</option>
                                        <option value="0">Rural</option>
                                    </select>
                                </div>
                                <!-- Submit Button -->
                                <div class="col-12 mt-4">
                                    <button type="submit" class="btn btn-custom w-100">Predict Loan Eligibility</button>
                                </div>
                            </div>
                        </form>

                        {% if prediction_text %}
                            <div class="result-box {{ 'approved' if 'Approved' in prediction_text else 'rejected' }}">
                                {{ prediction_text }}
                            </div>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    if request.method == 'POST':
        try:
            # Extract form values in the exact order expected by feature_names_in_
            features = [
                float(request.form['Gender']),
                float(request.form['Married']),
                float(request.form['Dependents']),
                float(request.form['Education']),
                float(request.form['Self_Employed']),
                float(request.form['ApplicantIncome']),
                float(request.form['CoapplicantIncome']),
                float(request.form['LoanAmount']),
                float(request.form['Loan_Amount_Term']),
                float(request.form['Credit_History']),
                float(request.form['Property_Area'])
            ]
            
            # Reshape for prediction
            final_features = np.array([features])
            prediction = model.predict(final_features)
            
            # Assuming 1 = Approved, 0 = Rejected
            if prediction[0] == 1:
                prediction_text = "Congratulations! Loan Status: APPROVED"
            else:
                prediction_text = "Sorry, Loan Status: REJECTED"
                
        except Exception as e:
            prediction_text = f"Error during prediction: {str5(e)}"

    return render_template_string(HTML_TEMPLATE, prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
