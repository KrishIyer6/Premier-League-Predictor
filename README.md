# Premier League Match Outcome Prediction

This repository contains a comprehensive machine learning pipeline for predicting Premier League match outcomes (Home Win, Away Win, or Draw) using historical match data. The project uses Python with scikit-learn for model training, focusing on Random Forest and Gradient Boosting classifiers. It includes data preprocessing, feature engineering, model evaluation, and visualizations.


## Project Overview


- **Goal**: Build and evaluate machine learning models to predict the outcome of Premier League matches based on historical data.
- **Dataset**: `premier_league_data.csv` – Contains over 33,000 matches with details like team names, scores, attendance, referees, stadiums, seasons, and dates.
- **Key Steps**:
  - Data loading and cleaning (handling invalid scores, missing values).
  - Target variable creation (Match_Outcome: Home_Win, Away_Win, Draw).
  - Feature engineering (team performance stats, form streaks, head-to-head records, etc.).
  - Model training with hyperparameter tuning via GridSearchCV.
  - Evaluation using accuracy, F1-score, confusion matrices, and cross-validation.
  - Visualizations for outcome distribution, model comparisons, and feature importances.
- **Models Used**:
  - Random Forest Classifier
  - Gradient Boosting Classifier
- **Results**: 
  - Random Forest achieved ~XX% accuracy on test set (replace with actual from notebook run).
  - Gradient Boosting achieved ~XX% accuracy on test set.
  - Detailed metrics and confusion matrices are generated in the notebook.


## Requirements


- Python 3.9+
- Jupyter Notebook (for running `final.ipynb`)


Required libraries (install via `pip`):
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```


You can install them all at once:
```
pip install pandas numpy matplotlib seaborn scikit-learn
```


## Installation


1. Clone the repository:
   ```
   git clone https://github.com/yourusername/premier-league-prediction.git
   cd premier-league-prediction
   ```


2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   (If `requirements.txt` is not present, use the list above.)


3. Ensure the dataset `premier_league_data.csv` is in the root directory.


## Usage


1. Launch Jupyter Notebook:
   ```
   jupyter notebook
   ```


2. Open and run `final.ipynb` cell by cell.
   - The notebook loads the data, processes it, trains models, and generates visualizations.
   - Outputs include printed statistics, classification reports, and saved images like `model_comparison.png`.


3. To reproduce results:
   - Ensure the random seed is consistent (set in the notebook).
   - Run the entire notebook for end-to-end execution.


## Data


- **Source**: `premier_league_data.csv` (assumed scraped or collected from public sources like Premier League APIs or datasets).
- **Columns**: Home_team_score, Home_team_name, Away_team_score, Away_team_name, attendance, referee, stadium, Season, Date.
- **Preprocessing**: Scores converted to numeric, invalid rows removed, match outcomes derived.
- **Statistics**:
  - Total Matches: 33,227
  - Outcome Distribution: Home Win (44%), Away Win (33%), Draw (23%)


Note: If the dataset is large or proprietary, consider adding a note on how to obtain it.


## Results and Visualizations


- **Model Performance**:
  - Accuracy and F1-scores are computed for both models.
  - Confusion matrices show prediction breakdowns.
- **Feature Importances**: Highlight key predictors like team form, goal differences, and head-to-head stats.
- **Visuals**: Bar charts for accuracy/F1 comparisons, heatmaps for confusion matrices, and more (saved as PNG).


Example output from notebook:
```
Match Outcome Distribution:
Home_Win    14675
Away_Win    11015
Draw         7537
```


## Contributing


Contributions are welcome! Please fork the repo and submit a pull request. Areas for improvement:
- Add more advanced models (e.g., XGBoost, Neural Networks).
- Incorporate real-time data fetching.
- Enhance feature engineering (e.g., player stats, weather).


## Acknowledgments


- Built with scikit-learn and Jupyter.
- Inspired by sports analytics and machine learning for predictions.
