# Cricket Match Data Analysis and Prediction Using Machine Learning

## Introduction
Cricket is a sport influenced by multiple factors such as player performance, match conditions, and game dynamics. This project aims to analyze cricket match data and develop machine learning models to predict various aspects of the game, including match outcomes, player performance, runs per over, and wicket probabilities. By leveraging data-driven approaches, this project provides valuable insights and enhances predictive accuracy in cricket analytics.

## Features
- Data preprocessing and cleaning
- Data visualization to analyze match performance
- Feature engineering for improved predictions
- Machine learning models for predicting:
  - Match outcomes
  - Runs scored per over
  - Likelihood of wickets falling
- Performance evaluation of models

## Installation
To run this project, install the required dependencies using the following:
```bash
pip install -r requirements.txt
```

## Usage
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Cricket_Match_Data_Analysis_and_Prediction_Using_Machine_Learning.ipynb
   ```
2. Run the notebook cells sequentially to process data, visualize insights, and train machine learning models.
3. Adjust hyperparameters as needed to improve model accuracy.

## Streamlit Web Application
A modern Streamlit web application is included for interactive data exploration and prediction.

### How to Run the App
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```
3. Use the sidebar and dropdowns to:
   - Preview and clean data
   - Select and view interactive visualizations
   - Train a simple logistic regression model
   - Predict match outcomes based on user input

### Features in the App
- Data preview and cleaning
- Selectable visualizations:
  - Runs distribution
  - Team performance
  - Player performance (Top 10 batsmen)
  - Top venues for total runs scored
  - Winner counts by country
- Model training and prediction UI
- Insights and conclusion sections
- Author and links footer

## Data Preprocessing
The dataset is cleaned and prepared before analysis. The preprocessing steps include:
- Handling missing values
- Encoding categorical data
- Feature selection and transformation

## Data Visualization
The notebook includes various visualizations such as:
- Runs distribution
- Team performance comparison
- Player performance analysis
- Top venues for total runs scored

## Machine Learning Models Used
- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting
- Support Vector Machines (SVM)

## Evaluation Metrics
- Accuracy
- Precision & Recall
- Confusion Matrix
- ROC Curve

## Future Enhancements
- Integration of real-time match data
- Advanced deep learning models
- Incorporation of weather and pitch conditions

## Contributing
Feel free to fork this repository, make improvements, and submit pull requests.

## License
This project is licensed under the MIT License.

## Data Files
- `deliveries.csv`: Ball-by-ball delivery data
- `matches.csv`: Match-level data (including venue, winner, etc.)

## Author
Made by **Yameen Munir**  
[GitHub](https://github.com/YameenMunir) | [LinkedIn](https://www.linkedin.com/in/yameen-munir/) | [Portfolio](https://www.datascienceportfol.io/YameenMunir)

