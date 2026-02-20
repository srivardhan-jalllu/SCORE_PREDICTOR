# Cricket Score Predictor 🏏

A Machine Learning project that predicts cricket match scores using Random Forest Regression algorithm.

## 📊 Project Overview

This project uses historical IPL match data to predict the final score of a cricket match based on the current match state. The model achieves **67.34% R-squared score** and **65.27% accuracy** within ±10 runs.

## 🎯 Features Used

1. **runs** - Current total runs scored
2. **wickets** - Current wickets fallen
3. **overs** - Current overs bowled
4. **striker** - Striker's individual runs
5. **non-striker** - Non-striker's individual runs

## 📈 Model Performance

- **R-squared Score**: 67.34%
- **Custom Accuracy (±10 runs)**: 65.27%
- **Custom Accuracy (±20 runs)**: 83.70%

## 🚀 How to Run

### Option 1: Google Colab (Recommended) 🌐

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **Upload** → Upload `Cricket_Score_Predictor.ipynb`
3. Upload `ipl.csv` file (click 📁 folder icon → Upload)
4. Run all cells: **Runtime** → **Run all** (or press `Shift + Enter` in each cell)
5. Done! 🎉

### Option 2: Python Script 💻

```bash
# Install dependencies
pip install pandas scikit-learn numpy matplotlib seaborn

# Run the model
python3 cricket_predictor.py

# Make a custom prediction
python3 cricket_predictor.py --predict 100 0 13 50 50
```

### Option 3: Local Jupyter Notebook 📓

```bash
# Install Jupyter
pip install jupyter

# Launch Jupyter
jupyter notebook

# Open Cricket_Score_Predictor.ipynb
```

## 📁 Project Files

### Main Files
- `Cricket_Score_Predictor.ipynb` - Jupyter notebook for Google Colab
- `cricket_predictor.py` - Python script for local execution
- `README.md` - This file

### Datasets
- `ipl.csv` - IPL matches dataset (76,014 records) ⭐
- `odi.csv` - ODI matches dataset (22,658 records)
- `t20.csv` - T20 matches dataset (19,968 records)

## 💡 Example Predictions

| Match State | Current Score | Wickets | Overs | Predicted Score |
|-------------|---------------|---------|-------|----------------|
| Good start | 100 runs | 0 | 13 | 176.4 runs |
| Moderate | 60 runs | 3 | 10 | 149.8 runs |
| Slow start | 40 runs | 5 | 8 | 78.8 runs |
| Great position | 150 runs | 1 | 15 | 214.0 runs |

## 🔧 Customize Predictions

In the notebook, edit **Step 10** to change these values:
```python
custom_runs = 120       # Current runs
custom_wickets = 2      # Current wickets
custom_overs = 12       # Current overs
custom_striker = 60     # Striker's runs
custom_non_striker = 50 # Non-striker's runs
```

## 📚 How the Model Works

1. **Data Loading**: Loads ball-by-ball IPL match data
2. **Feature Extraction**: Extracts 5 key features
3. **Data Splitting**: 75% training, 25% testing
4. **Feature Scaling**: Normalizes features using StandardScaler
5. **Model Training**: Trains Random Forest Regression (100 trees)
6. **Evaluation**: Calculates R² score and custom accuracy
7. **Prediction**: Predicts final score based on current match state

## 🎓 Key Insights

- Random Forest outperforms Linear Regression for this problem
- Current runs and wickets are the strongest predictors
- Model performs well despite cricket's inherent unpredictability
- Custom accuracy metric is more intuitive than R² for cricket fans

## 🔮 Future Improvements

- Add more features (venue, team strength, player form)
- Try deep learning models (LSTM/RNN for sequential data)
- Include real-time match updates
- Build a web interface for live predictions

## 📝 Requirements

```
pandas>=2.0.0
scikit-learn>=1.0.0
numpy>=1.20.0
matplotlib>=3.0.0
seaborn>=0.11.0
```

## 🤝 Contributing

Feel free to fork this project and make improvements!

## 📄 License

This project is open source and available for educational purposes.

---

**🎉 Happy Predicting!** 🏏
