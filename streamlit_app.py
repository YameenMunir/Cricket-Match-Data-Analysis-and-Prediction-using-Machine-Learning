import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load, dump
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

st.title("🏏 Cricket Match Data Analysis and Prediction Using Machine Learning")

st.header("📖 Introduction")
st.write("""
Cricket is a sport which consists of factors such as influencing match outcomes, player performance, and game dynamics. This project aims to analyze cricket match data and develop some machine learning models to predict various aspects of the game, including match outcomes, player performance, runs scored per over, and the likelihood of wickets falling. By leveraging data-driven approaches, I aim to provide valuable insights and improve predictive accuracy in cricket analytics.
""")

# Data Preprocessing
st.header("🔧 Data Preprocessing")
@st.cache_data(show_spinner=False)
def load_data():
    data1 = pd.read_csv('deliveries.csv')
    data2 = pd.read_csv('matches.csv')
    return data1, data2

data1, data2 = load_data()

st.subheader("📊 First 5 rows of deliveries data")
st.dataframe(data1.head())

st.subheader("📈 Last 5 rows of deliveries data")
st.dataframe(data1.tail())

# Data Cleaning (as in notebook)
categorical_columns_with_missing_values = ['wicket_type', 'player_dismissed']
for col in categorical_columns_with_missing_values:
    if col in data1.columns:
        data1[col] = data1[col].fillna('None')
categorical_columns_with_missing_values2 = ['extras', 'player_dismissed']
for col in categorical_columns_with_missing_values2:
    if col in data1.columns:
        data1[col] = data1[col].fillna('0')
columns_to_drop = [
    'wides', 'noballs', 'byes', 'legbyes', 'penalty',
    'wicket_type', 'player_dismissed', 'other_wicket_type', 'other_player_dismissed'
]
data1 = data1.drop(columns=[col for col in columns_to_drop if col in data1.columns])
st.subheader("🔍 Null values in deliveries data")
st.write(data1.isnull().sum())

# Data Visualization (as in notebook)
st.header("📈 Data Visualization")

viz_options = [
    "Runs Distribution",
    "Team Performance",
    "Player Performance (Top 10 Batsmen)",
    "Total Runs Scored at Top 10 Venues",
    "Winner Counts by Country (Top 10)"
]
viz_choice = st.selectbox("Select a visualization to display:", viz_options)

if viz_choice == "Runs Distribution":
    st.subheader("📊 1. Runs Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data1['batsman_runs'], bins=30, kde=True, ax=ax)
    ax.set_title('Distribution of Runs Scored')
    ax.set_xlabel('Runs')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
elif viz_choice == "Team Performance":
    st.subheader("🏆 2. Team Performance")
    if 'batting_team' in data1.columns:
        team_performance = data1.groupby('batting_team')['batsman_runs'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='batsman_runs', y='batting_team', data=team_performance, palette='viridis', ax=ax)
        ax.set_title('Team Performance: Total Runs Scored')
        ax.set_xlabel('Total Runs')
        ax.set_ylabel('Team')
        st.pyplot(fig)
elif viz_choice == "Player Performance (Top 10 Batsmen)":
    st.subheader("🏏 3. Player Performance (Top 10 Batsmen)")
    if 'batsman' in data1.columns:
        player_performance = data1.groupby('batsman')['batsman_runs'].sum().reset_index()
        top_batsmen = player_performance.sort_values(by='batsman_runs', ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='batsman_runs', y='batsman', data=top_batsmen, palette='magma', ax=ax)
        ax.set_title('Top 10 Batsmen: Total Runs Scored')
        ax.set_xlabel('Total Runs')
        ax.set_ylabel('Player')
        st.pyplot(fig)
elif viz_choice == "Total Runs Scored at Top 10 Venues":
    st.subheader("🏟️ 4. Total Runs Scored at Top 10 Venues")
    # Merge deliveries and matches to get venue info
    if 'match_id' in data1.columns and 'id' in data2.columns and 'venue' in data2.columns:
        merged = pd.merge(data1, data2[['id', 'venue']], left_on='match_id', right_on='id', how='left')
        venue_performance = merged.dropna(subset=['venue']).groupby('venue')['batsman_runs'].sum().reset_index()
        if not venue_performance.empty:
            top_venues = venue_performance.sort_values(by='batsman_runs', ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(x='batsman_runs', y='venue', data=top_venues, palette='magma', ax=ax)
            ax.set_title('Top 10 Venues: Total Runs Scored')
            ax.set_xlabel('Total Runs')
            ax.set_ylabel('Venue')
            st.pyplot(fig)
        else:
            st.warning("No venue data available for visualization.")
    else:
        st.warning("Could not merge deliveries and matches to get venue information.")
elif viz_choice == "Winner Counts by Country (Top 10)":
    st.subheader("🥇 Winner Counts by Country (Top 10)")
    if 'winner' in data2.columns:
        winner_counts = data2['winner'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(21, 8))
        sns.barplot(x=winner_counts.index, y=winner_counts.values, palette='magma', ax=ax)
        ax.set_title('Winner Counts by Country')
        ax.set_xlabel('Country')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Machine Learning Model (simple version, as in notebook)
st.header("🤖 Machine Learning Prediction Model")
st.write("""
A logistic regression model is used to predict match outcomes based on features such as batsman_runs and day_of_week. The model is trained and saved as 'cricket_match_predictor.joblib'.
""")

# --- Optimization: Cache model training ---
@st.cache_resource(show_spinner=False)
def train_simple_model(data1, data2):
    if 'start_date' in data2.columns and 'match_id' in data1.columns and 'id' in data2.columns:
        merged = pd.merge(data1, data2[['id', 'start_date']], left_on='match_id', right_on='id', how='left')
        merged['day_of_week'] = pd.to_datetime(merged['start_date'], errors='coerce').dt.dayofweek.fillna(0).astype(int)
    else:
        merged = data1.copy()
        merged['day_of_week'] = 0
    merged['match_outcome'] = (merged['batsman_runs'] > 4).astype(int)
    X = merged[['batsman_runs', 'day_of_week']]
    y = merged['match_outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(X_scaled, y)
    return model, scaler

st.header("🎯 Train Logistic Regression Model (Simple Features)")
st.write("""
This will train a new model using only 'batsman_runs' and 'day_of_week' as features, so the prediction UI will work correctly.
""")
if st.button("Train Simple Model"):
    try:
        model, scaler = train_simple_model(data1, data2)
        st.success("Simple model trained and cached. Now the prediction UI will work.")
        # Optionally save model
        dump(model, 'cricket_match_predictor.joblib')
    except Exception as model_error:
        st.error(f"Simple model training failed: {model_error}")

# Prediction Section
st.header("🔮 Prediction Section")
st.subheader("🎲 Predict Match Outcome")

# Remove duplicate data loading
if 'match_id' in data1.columns and 'id' in data2.columns:
    data = pd.merge(data1, data2, left_on='match_id', right_on='id')
else:
    st.warning("Could not merge deliveries and matches on 'match_id' and 'id'. Using deliveries only.")
    data = data1.copy()
if 'start_date' in data.columns:
    data['day_of_week'] = pd.to_datetime(data['start_date']).dt.dayofweek
else:
    data['day_of_week'] = 0
runs = st.number_input('Batsman Runs', min_value=0, max_value=50, value=10)
day_of_week = st.selectbox('Day of Week', options=[0,1,2,3,4,5,6], format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
features = pd.DataFrame({'batsman_runs': [runs], 'day_of_week': [day_of_week]})
try:
    model, scaler = train_simple_model(data1, data2)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    st.write(f"Predicted Match Outcome: {'Likely Win' if prediction == 1 else 'Likely Not Win'}")
except Exception as e:
    st.warning(f"Model not found or error in prediction: {e}")

# Add notebook-style insights as markdown
st.header("💡 Insights")
st.markdown("""
- The target variable has two possible: 0 and 1, likely representing different outcomes of cricket match (in this case win or loss).
- It is clear that there is imbalance with the distribution, with approximately 80% of the data points having a target value of 0.
- The high frequency of 0 and 1 suggests that a large proportion of deliveries result in either no runs or a single run being scored.
- The long tail of the distribution indicates that there are occasional deliveries where a large number of runs are scored, which can be crucial in determining the outcome of a match.
- The imbalanced distribution of the target variable may pose challenges for model training and evaluation.
- Overall, the data suggests that scoring runs in cricket is a challenging task, with a high probability of scoring few runs on any given delivery. However, the occasional 4 or 6 can be a game changer in increasing the chances of a particular team from winning (depending on the consistency at which 4s and 6s are scored throughout overs).
""")

st.header("📝 Conclusion")
st.write("""
Overall, the data suggests that scoring runs in cricket is a challenging task, with a high probability of scoring few runs on any given delivery. However, the occasional 4 or 6 can be a game changer in increasing the chances of a particular team from winning (depending on the consistency at which 4s and 6s are scored throughout overs).
""")

st.markdown('---')
st.markdown(
    '<div style="text-align:center; font-size:1.1em; padding: 20px;">'
    '    Made By <b>👨‍💻 Yameen Munir</b><br><br>'
    '    <a href="https://github.com/YameenMunir" target="_blank" style="text-decoration:none; margin:0 10px;">'
    '        🐙 GitHub</a> &nbsp;|&nbsp; '
    '    <a href="https://www.linkedin.com/in/yameen-munir/" target="_blank" style="text-decoration:none; margin:0 10px;">'
    '        💼 LinkedIn</a> &nbsp;|&nbsp; '
    '    <a href="https://yameenmunir.vercel.app/" target="_blank" style="text-decoration:none; margin:0 10px;">'
    '        🌐 Website</a>'
    '</div>',
    unsafe_allow_html=True
)
