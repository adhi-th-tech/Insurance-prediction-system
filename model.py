# %%
import pandas as pd
import numpy as np
import joblib


from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

# %%
df = pd.read_csv('medical_insurance.csv')

# %%
# Clean column names and object data
df.columns = df.columns.str.strip()

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str).str.strip()

# %%
# Premium is strongly related to expenses, so we remove it to avoid data leakage
df = df.drop(columns=['premium'])

# %%
# Drop duplicates
df = df.drop_duplicates()

# %%
# Encode categorical variables
ordinal_encoder = OrdinalEncoder()

df[['gender', 'discount_eligibility']] = ordinal_encoder.fit_transform(
    df[['gender', 'discount_eligibility']]
)

# %%
# Create dummy variables for region
df = pd.get_dummies(df, columns=['region'], drop_first=True)

# ==========================================
# CLASSIFICATION MODEL (Discount Eligibility)
# ==========================================
# %%
# We must drop BOTH the discount and the expenses so the model 
# only learns from the raw user inputs (age, gender, bmi, etc.)
feature_cols = [col for col in df.columns if col not in ['discount_eligibility', 'expenses']]

X_rf = df[feature_cols]
y_rf = df['discount_eligibility']

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

# FIX: Rename to scaler_class
scaler_class = StandardScaler()
X_train_scaled_class = scaler_class.fit_transform(X_train_rf)
X_test_scaled_class = scaler_class.transform(X_test_rf)

model_svc = SVC(kernel='rbf', random_state=42)
model_svc.fit(X_train_scaled_class, y_train_rf)
predictions_svc = model_svc.predict(X_test_scaled_class)

# ==========================================
# REGRESSION MODEL (Expenses) - FIXED LINEAR
# ==========================================
X = df.drop(columns=['expenses'])
y = df['expenses']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Initialize and fit the scaler
scaler_reg = StandardScaler()
X_train_scaled_reg = scaler_reg.fit_transform(X_train)
X_test_scaled_reg = scaler_reg.transform(X_test)

# Initialize Linear Regression
linear_model = LinearRegression()

# FIX: Train the model on the SCALED data (X_train_scaled_reg), not the raw X_train
linear_model.fit(X_train_scaled_reg, y_train)

# Test the model
y_pred_lr = linear_model.predict(X_test_scaled_reg)

# Save the Linear Regression model and the matching scaler
joblib.dump(linear_model, 'expense_lr_model.pkl')
joblib.dump(scaler_reg, 'scaler_reg.pkl')


# 2. Save your Classification Model (Predicts Discount Eligibility)
joblib.dump(model_svc, 'discount_svm_model.pkl')

# 3. Save your preprocessors
# The classification model requires scaled data, so we save the scaler_class.
joblib.dump(scaler_class, 'scaler_class.pkl')
joblib.dump(ordinal_encoder, 'ordinal_encoder.pkl')

print("\nModels and preprocessors successfully saved to binary files!")