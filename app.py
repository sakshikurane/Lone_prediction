        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self-employed", ["No", "Yes"])
        property_area = st.selectbox("Property area", ["Rural", "Semiurban", "Urban"])

    with col2:
        applicant_income = st.number_input("Applicant income", min_value=0.0, value=5000.0, step=500.0)
        coapplicant_income = st.number_input("Co-applicant income", min_value=0.0, value=0.0, step=500.0)
        loan_amount = st.number_input("Loan amount", min_value=0.0, value=120.0, step=10.0,
                                      help="Use the same units as the training dataset.")
        loan_term = st.number_input("Loan amount term", min_value=1.0, value=360.0, step=12.0)
        credit_history = st.selectbox("Credit history", ["Good (1)", "Poor (0)"])

    submitted = st.form_submit_button("Predict loan outcome", type="primary")

if submitted:
    # Typical alphabetical LabelEncoder mappings for these categories.
    encoded = {
        "Gender": {"Female": 0, "Male": 1}[gender],
        "Married": {"No": 0, "Yes": 1}[married],
        "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3}[dependents],
        "Education": {"Graduate": 0, "Not Graduate": 1}[education],
        "Self_Employed": {"No": 0, "Yes": 1}[self_employed],
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": {"Good (1)": 1, "Poor (0)": 0}[credit_history],
        "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area],
    }

    feature_names = getattr(model, "feature_names_in_", list(encoded.keys()))
    input_df = pd.DataFrame([[encoded[name] for name in feature_names]], columns=feature_names)

    try:
        prediction = int(model.predict(input_df)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)
            if 1 in classes:
                probability = float(probabilities[classes.index(1)])

        if prediction == 1:
            st.success("Prediction: Loan likely to be approved (class 1).")
        else:
            st.warning("Prediction: Loan likely not to be approved (class 0).")

        if probability is not None:
            st.metric("Model probability for class 1", f"{probability:.1%}")

        st.caption(
            "Important: this result depends on the category encodings and target labels used "
            "during training. Verify those mappings before relying on predictions."
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info("Check that the input feature names, order, data types, and category encodings match training.")
