#!/usr/bin/env python3
"""
Cricket Score Predictor
Uses Machine Learning to predict cricket match scores based on current match state.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import sys

def custom_accuracy(y_test, y_pred, threshold):
    """
    Calculate custom accuracy based on how close predictions are to actual values.
    
    Args:
        y_test: Actual values
        y_pred: Predicted values
        threshold: Acceptable difference between predicted and actual
    
    Returns:
        Accuracy percentage
    """
    right = 0
    l = len(y_pred)
    for i in range(0, l):
        if abs(y_pred[i] - y_test[i]) <= threshold:
            right += 1
    return (right/l) * 100

def train_model(csv_file='ipl.csv', test_size=0.25, random_state=0):
    """
    Train the cricket score prediction model.
    
    Args:
        csv_file: Path to the dataset CSV file
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        model, scaler, test_data
    """
    print('=' * 70)
    print('Cricket Score Predictor - Machine Learning Model')
    print('=' * 70)
    
    # Load dataset
    print(f'\n📊 Loading dataset from {csv_file}...')
    dataset = pd.read_csv(csv_file)
    print(f'   Dataset loaded: {len(dataset):,} rows')
    print(f'   Dataset shape: {dataset.shape}')
    
    # Extract features and labels
    print('\n🔍 Extracting features and labels...')
    X = dataset.iloc[:,[7,8,9,12,13]].values  # runs, wickets, overs, striker, non-striker
    y = dataset.iloc[:, 14].values  # total score
    
    print(f'   Features used: runs, wickets, overs, striker, non-striker')
    print(f'   Feature shape: {X.shape}')
    print(f'   Label shape: {y.shape}')
    
    # Split data
    print(f'\n✂️  Splitting data into train/test sets ({int((1-test_size)*100)}/{int(test_size*100)})...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f'   Training set: {X_train.shape[0]:,} samples')
    print(f'   Testing set: {X_test.shape[0]:,} samples')
    
    # Feature scaling
    print('\n📏 Applying feature scaling...')
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    print('   Feature scaling completed')
    
    # Train Random Forest model
    print('\n🤖 Training Random Forest Regression Model...')
    print('   (This may take a few moments...)')
    model = RandomForestRegressor(n_estimators=100, max_features=None, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    print('   ✓ Model training completed!')
    
    # Evaluate model
    print('\n📈 Evaluating model performance...')
    score = model.score(X_test_scaled, y_test) * 100
    print(f'   R-squared value: {score:.2f}%')
    
    # Custom accuracy
    y_pred = model.predict(X_test_scaled)
    accuracy_10 = custom_accuracy(y_test, y_pred, 10)
    accuracy_20 = custom_accuracy(y_test, y_pred, 20)
    
    print(f'   Custom accuracy (±10 runs): {accuracy_10:.2f}%')
    print(f'   Custom accuracy (±20 runs): {accuracy_20:.2f}%')
    
    print('\n' + '=' * 70)
    print('Model Training Complete!')
    print('=' * 70)
    
    return model, sc, (X_test, y_test, y_pred)

def predict_score(model, scaler, runs, wickets, overs, striker, non_striker):
    """
    Predict the final score based on current match state.
    
    Args:
        model: Trained Random Forest model
        scaler: Fitted StandardScaler
        runs: Current runs scored
        wickets: Current wickets fallen
        overs: Current overs bowled
        striker: Runs scored by striker
        non_striker: Runs scored by non-striker
    
    Returns:
        Predicted final score
    """
    input_data = np.array([[runs, wickets, overs, striker, non_striker]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return prediction[0]

def main():
    """Main function to run the cricket score predictor."""
    
    # Train the model
    model, scaler, test_data = train_model('ipl.csv')
    
    # Example predictions
    print('\n' + '=' * 70)
    print('Example Predictions')
    print('=' * 70)
    
    examples = [
        (100, 0, 13, 50, 50, "Good start, no wickets"),
        (60, 3, 10, 30, 20, "Moderate start, some wickets"),
        (40, 5, 8, 15, 15, "Slow start, many wickets"),
        (150, 1, 15, 80, 60, "Great position"),
    ]
    
    for runs, wickets, overs, striker, non_striker, desc in examples:
        predicted = predict_score(model, scaler, runs, wickets, overs, striker, non_striker)
        print(f'\n📊 Match State: {desc}')
        print(f'   Current: {runs} runs, {wickets} wickets, {overs} overs')
        print(f'   Batsmen: Striker={striker}, Non-striker={non_striker}')
        print(f'   ➜ Predicted Final Score: {predicted:.1f} runs')
    
    # Display final accuracy summary
    print('\n' + '=' * 70)
    print('📊 Final Model Accuracy Summary')
    print('=' * 70)
    
    X_test, y_test, y_pred = test_data
    r2_score = model.score(scaler.transform(X_test), y_test) * 100
    acc_10 = custom_accuracy(y_test, y_pred, 10)
    acc_20 = custom_accuracy(y_test, y_pred, 20)
    errors = y_pred - y_test
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    print(f'\n🎯 R-squared Score: {r2_score:.2f}%')
    print(f'✅ Custom Accuracy (±10 runs): {acc_10:.2f}%')
    print(f'✅ Custom Accuracy (±20 runs): {acc_20:.2f}%')
    print(f'📉 Mean Error: {mean_error:.2f} runs')
    print(f'📉 Standard Deviation: {std_error:.2f} runs')
    
    print('\n' + '=' * 70)
    print('Interactive Mode')
    print('=' * 70)
    print('\nYou can now predict scores for any match state!')
    print('Run: python cricket_predictor.py --predict <runs> <wickets> <overs> <striker> <non_striker>')
    print('=' * 70)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--predict':
        if len(sys.argv) != 7:
            print("Usage: python cricket_predictor.py --predict <runs> <wickets> <overs> <striker> <non_striker>")
            sys.exit(1)
        
        runs = float(sys.argv[2])
        wickets = float(sys.argv[3])
        overs = float(sys.argv[4])
        striker = float(sys.argv[5])
        non_striker = float(sys.argv[6])
        
        model, scaler, test_data = train_model('ipl.csv')
        predicted = predict_score(model, scaler, runs, wickets, overs, striker, non_striker)
        
        print('\n' + '=' * 70)
        print('🎯 Prediction Result')
        print('=' * 70)
        print(f'\nMatch State:')
        print(f'  • Runs: {runs}')
        print(f'  • Wickets: {wickets}')
        print(f'  • Overs: {overs}')
        print(f'  • Striker: {striker}')
        print(f'  • Non-striker: {non_striker}')
        print(f'\n➜ Predicted Final Score: {predicted:.1f} runs')
        
        # Display accuracy summary
        print('\n' + '=' * 70)
        print('📊 Model Accuracy Summary')
        print('=' * 70)
        
        X_test, y_test, y_pred = test_data
        r2_score = model.score(scaler.transform(X_test), y_test) * 100
        acc_10 = custom_accuracy(y_test, y_pred, 10)
        acc_20 = custom_accuracy(y_test, y_pred, 20)
        errors = y_pred - y_test
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        print(f'\n🎯 R-squared Score: {r2_score:.2f}%')
        print(f'✅ Custom Accuracy (±10 runs): {acc_10:.2f}%')
        print(f'✅ Custom Accuracy (±20 runs): {acc_20:.2f}%')
        print(f'📉 Mean Error: {mean_error:.2f} runs')
        print(f'📉 Standard Deviation: {std_error:.2f} runs')
        print('=' * 70)
    else:
        main()

