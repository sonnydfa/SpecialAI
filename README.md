data = pd.read_csv('PRSA_TRAIN.csv')


data['pm2.5'].isnull().sum()
data.dropna(subset=['pm2.5'], inplace=True)
data['pm2.5'].isnull().sum()
# Perform one-hot encoding on the 'cbwd' column
data_encoded = pd.get_dummies(data, columns=['cbwd'])

data = data_encoded
# Prepare the data for modeling
X = data.drop(columns=['pm2.5'])
y = data['pm2.5']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Make predictions
y_pred = rf_regressor.predict(X_test)

# Calculate the Mean Squared Error, RMSE and R^2 score
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared (R²) Score: {r2}')


# Save the trained model to a file
joblib.dump(rf_regressor, 'rf_regressor_model.pkl')

# Load the test data
test_data = pd.read_csv('PRSA_TEST.csv')
data_encoded = pd.get_dummies(test_data, columns=['cbwd'])

test_data = data_encoded
# Make predictions on the test data
test_predictions = rf_regressor.predict(test_data)

# Append the predictions to the test data
test_data['Predicted_PRSA_TEST'] = test_predictions

# Save the results to a new CSV file
test_data.to_csv('TEST_with_predictions.csv', index=False)
