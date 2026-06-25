# %%
import pandas as pd
import numpy as np
import joblib


from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score, 
    accuracy_score, 
    classification_report
)

# %%
df = pd.read_csv('medical_insurance_5870_correlation_repaired.csv')

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

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train_scaled_class, y_train_rf)

predictions_rf = model_rf.predict(X_test_scaled_class)

print(f"Random Forest Accuracy: {accuracy_score(y_test_rf, predictions_rf):.4f}")
print("\nClassification Report:\n", classification_report(y_test_rf, predictions_rf))

# %%
model_lr = LogisticRegression(random_state=42)
model_lr.fit(X_train_scaled_class, y_train_rf)
predictions_lr = model_lr.predict(X_test_scaled_class)

# %%
model_svc = SVC(kernel='rbf', random_state=42)
model_svc.fit(X_train_scaled_class, y_train_rf)
predictions_svc = model_svc.predict(X_test_scaled_class)

# ==========================================
# REGRESSION MODEL (Expenses)
# ==========================================
# %%
X = df.drop(columns=['expenses'])
y = df['expenses']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# FIX: Rename to scaler_reg
scaler_reg = StandardScaler()
X_train_scaled_reg = scaler_reg.fit_transform(X_train)
X_test_scaled_reg = scaler_reg.transform(X_test)

# %%
linear_model = LinearRegression()
linear_model.fit(X_train_scaled_reg, y_train)

y_pred_lr = linear_model.predict(X_test_scaled_reg)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\nLinear Regression Results")
print("R2 Score:", r2_lr)

# %%
tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

# %%
# Final chosen regression model: Random Forest
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# Tree-based models do not require scaled data
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Regressor Results")
print("R2 Score:", r2_score(y_test, y_pred_rf))

# ==========================================
# EXPORT MODELS FOR BACKEND
# ==========================================
# %%
# 1. Save your Regression Model (Predicts Expenses)
joblib.dump(rf, 'expense_rf_model.pkl')

# 2. Save your Classification Model (Predicts Discount Eligibility)
joblib.dump(model_rf, 'discount_rf_model.pkl')

# 3. Save your preprocessors
# The classification model requires scaled data, so we save the scaler_class.
# The regression model (Random Forest) did not use scaled data, so we don't need to save scaler_reg.
joblib.dump(scaler_class, 'scaler_class.pkl')
joblib.dump(ordinal_encoder, 'ordinal_encoder.pkl')

print("\nModels and preprocessors successfully saved to binary files!")