import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import random

# ================== LOAD DATA ================== #
st.set_page_config(page_title="Fake Account Detection", layout="wide")

st.title("🔍 Fake Account Detection System")
st.write("Analyze whether a social media account is **Fake or Genuine** using multiple Machine Learning models and Q-learning.")

df = pd.read_csv('train.csv')

# Features & Target
X = df.drop(columns=['fake'])
y = df['fake']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================== USER INPUT FORM ================== #
st.sidebar.header("🔹 Enter Account Details")

profile_pic = st.sidebar.radio("Profile Picture", [1, 0])
username_length = st.sidebar.number_input("Numbers/Length in Username", min_value=0.0, max_value=20.0, value=10.0)
fullname_words = st.sidebar.number_input("Fullname Words", min_value=0, max_value=5, value=2)
fullname_numbers = st.sidebar.number_input("Numbers/Length in Fullname", min_value=0.0, max_value=20.0, value=5.0)
name_equals_username = st.sidebar.radio("Name == Username", [1, 0])
description_length = st.sidebar.number_input("Description Length", min_value=0, max_value=100, value=10)
external_url = st.sidebar.radio("Has External URL", [1, 0])
private = st.sidebar.radio("Is Private", [1, 0])
num_posts = st.sidebar.number_input("# Posts", min_value=0, max_value=10000, value=100)
num_followers = st.sidebar.number_input("# Followers", min_value=0, max_value=1000000, value=500)
num_follows = st.sidebar.number_input("# Follows", min_value=0, max_value=1000000, value=300)

# Convert input into DataFrame
input_data = pd.DataFrame([[profile_pic, username_length, fullname_words, fullname_numbers,
                            name_equals_username, description_length, external_url, private,
                            num_posts, num_followers, num_follows]], columns=X.columns)

# Standardize Input
input_scaled = scaler.transform(input_data)

# ================== MACHINE LEARNING MODELS ================== #
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}

st.sidebar.subheader("🔹 Model Selection")
selected_model_name = st.sidebar.selectbox("Choose a Model for Prediction", list(models.keys()))

# Train Models & Store Accuracy
accuracy_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    accuracy_results[name] = accuracy

selected_model = models[selected_model_name]

# ================== MODEL PREDICTION ================== #
st.subheader("📌 Model Performance Overview")
st.write("Below are the accuracy scores of different ML models on the test dataset:")

for name, acc in accuracy_results.items():
    st.write(f"**{name}**: {acc:.2f}")

if st.button("🔍 Predict Account Type"):
    prediction = selected_model.predict(input_scaled)[0]
    
    if prediction == 0:
        st.success(f"✅ {selected_model_name} predicts this account as **GENUINE**.")
    else:
        st.error(f"⚠️ {selected_model_name} predicts this account as **FAKE**.")

# ================== PIE CHARTS FOR MODELS ================== #
st.subheader("📊 Fake vs. Genuine Accounts - Model Predictions")

show_pie_chart = st.checkbox("Show Pie Charts for All Models")

if show_pie_chart:
    cols = st.columns(len(models))

    for idx, (name, model) in enumerate(models.items()):
        y_pred_test = model.predict(X_test)
        fake_count = np.sum(y_pred_test == 1)
        genuine_count = np.sum(y_pred_test == 0)

        labels = ["Fake Accounts", "Genuine Accounts"]
        sizes = [fake_count, genuine_count]
        colors = ['red', 'green']

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.axis('equal')

        with cols[idx]:
            st.write(f"**{name}**")
            st.pyplot(fig)

# ================== Q-LEARNING IMPLEMENTATION ================== #
st.subheader("🔬 Q-Learning Algorithm")

# Q-learning Parameters
state_size = X_train.shape[0]  
action_size = 2  # Two possible actions: Fake (1) or Genuine (0)
q_table = np.zeros((state_size, action_size))

# Hyperparameters
learning_rate = 0.1
discount_factor = 0.9
epsilon = 1.0
epsilon_decay = 0.99
epsilon_min = 0.01
episodes = 1000

# Q-Learning Training Loop
for episode in range(episodes):
    state_index = np.random.randint(0, state_size)  # Select random state
    state = X_train[state_index]

    # Epsilon-Greedy Strategy
    if random.uniform(0, 1) < epsilon:
        action = np.random.choice([0, 1])  # Exploration
    else:
        action = np.argmax(q_table[state_index])  # Exploitation

    # Reward System
    reward = 1 if action == y_train.iloc[state_index] else -1

    # Q-Table Update
    q_table[state_index, action] = (1 - learning_rate) * q_table[state_index, action] + \
                                   learning_rate * (reward + discount_factor * np.max(q_table[state_index]))

    # Epsilon Decay
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Q-Learning Predictions
q_predictions = np.argmax(q_table, axis=1)
q_learning_accuracy = np.mean(q_predictions == y_train.to_numpy())

st.write(f"**Q-Learning Accuracy on Training Data:** {q_learning_accuracy:.2f}")

# ================== Q-LEARNING PIE CHART ================== #
st.subheader("📊 Q-Learning Fake vs Genuine Predictions")

q_fake_count = np.sum(q_predictions == 1)
q_genuine_count = np.sum(q_predictions == 0)

fig2, ax2 = plt.subplots()
ax2.pie([q_fake_count, q_genuine_count], labels=["Fake Accounts", "Genuine Accounts"], autopct='%1.1f%%',
        colors=['red', 'green'], startangle=90)
ax2.axis('equal')

st.pyplot(fig2)

st.link_button("Go to Sentiment Analysis", "https://sentiment-analysis-hack.streamlit.app/")


# ================== FOOTER ================== #
st.markdown(
    """
    <style>
    .footer {
        text-align: center;
        margin-top: 20px;
        font-size: 14px;
        color: gray;
    }
    </style>
    <div class="footer">
        Developed by Faraz.
    </div>
    """,
    unsafe_allow_html=True
)
