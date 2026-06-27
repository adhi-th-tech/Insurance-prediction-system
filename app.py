from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# 1. Load models and preprocessors into memory ONCE when the server starts
print("Loading machine learning models...")
expense_model = joblib.load('expense_lr_model.pkl')
discount_model = joblib.load('discount_svm_model.pkl')
scaler_class = joblib.load('scaler_class.pkl')
ordinal_encoder = joblib.load('ordinal_encoder.pkl')
print("Models loaded successfully!")

# 2. Route to serve your HTML page
@app.route('/')
def home():
    return render_template('index.html')

# 3. Route to handle the predictions (Called by your frontend JavaScript)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data sent from the frontend form
        data = request.get_json()
        
        # Convert it to a pandas DataFrame
        df_user = pd.DataFrame([data])
        
        # --- PREPROCESSING (Same as your test script) ---
        # A. Encode gender
        # A. Encode gender (Convert string to numeric float)
        try:
            # Find the numerical index of the gender string in the encoder's saved categories
            gender_str = str(df_user['gender'][0]).lower().strip()
            gender_index = list(ordinal_encoder.categories_[0]).index(gender_str)
            df_user['gender'] = float(gender_index)
        except ValueError:
            # Safe fallback: usually 'female' is 0.0 and 'male' is 1.0 based on alphabetical order
            df_user['gender'] = 0.0 if str(df_user['gender'][0]).lower().strip() == 'female' else 1.0
            
        # B. Handle Region Dummy Columns
        expected_regions = ['region_northwest', 'region_southeast', 'region_southwest']
        for col in expected_regions:
            df_user[col] = 1 if f"region_{data['region']}" == col else 0
            
        df_user = df_user.drop(columns=['region'])
        
        # Enforce column order
        final_columns = ['age', 'gender', 'bmi', 'children', 'region_northwest', 'region_southeast', 'region_southwest']
        df_user = df_user[final_columns]
        
       # --- MAKE PREDICTIONS ---
        
        # 1. Predict Discount FIRST (Classification - Requires Scaling)
        # We do this first because the expense model needs this answer!
        df_user_scaled = scaler_class.transform(df_user)
        predicted_discount = discount_model.predict(df_user_scaled)[0]
        
        # 2. Add the predicted discount into our dataframe
        df_user['discount_eligibility'] = predicted_discount
        
        # 3. Rearrange columns to match EXACTLY how the expense model was trained
        # (It expects discount_eligibility right after children)
        final_reg_columns = [
            'age', 'gender', 'bmi', 'children', 
            'discount_eligibility', 
            'region_northwest', 'region_southeast', 'region_southwest'
        ]
        df_user_reg = df_user[final_reg_columns]
        
        # 4. Predict Expense (Regression)
        predicted_expense = expense_model.predict(df_user_reg)[0]
        
        # --- SEND RESPONSE BACK TO FRONTEND ---
        return jsonify({
            'success': True,
            'predicted_expense': round(predicted_expense, 2),
            'discount_eligible': bool(predicted_discount == 1)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5501)