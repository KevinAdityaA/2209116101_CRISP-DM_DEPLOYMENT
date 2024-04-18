
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
@st.cache_data()
def load_data():
    url = "Data_Cleaning2.csv"
    df = pd.read_csv(url)
    return df

# Function to perform EDA
def perform_eda(data):

    # Display basic statistics
    st.subheader("Basic Statistics")
    st.write(data.describe())
    
    # Display missing values
    st.subheader("Missing Values")
    st.write(data.isnull().sum())

    # Visualize general distributions of data
    st.subheader("General Distributions")
    # General explanation about wealthy individuals
    st.write("Analisis data tentang orang kaya bertujuan untuk memberikan pemahaman mendalam tentang karakteristik, sumber kekayaan, pola investasi, dan gaya hidup mereka melalui visualisasi histogram, statistik dasar, dan deteksi outlier. Dengan fokus pada distribusi kekayaan, analisis ini mengungkapkan pola umum tentang bagaimana kekayaan terbagi di antara individu, serta memungkinkan identifikasi sumber kekayaan, strategi investasi, dan preferensi gaya hidup yang umum digunakan. Selain itu, analisis membantu dalam memahami perbedaan distribusi kekayaan berdasarkan faktor-faktor seperti gender dan latar belakang sosial-ekonomi, memberikan wawasan yang luas tentang dinamika kekayaan dalam masyarakat.")
    for col in data.columns:
        if data[col].dtype in ['float64', 'int64']:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(data[col], kde=True, ax=ax)
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

    # Visualize distributions
    st.subheader("Wealth Distribution by Gender")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data, x='finalWorth', hue='gender', kde=True, ax=ax)
    ax.set_xlabel("Final Worth")
    ax.set_ylabel("Frequency")
    ax.set_title("Wealth Distribution by Gender")
    st.pyplot(fig)

                
# Function to preprocess data and train the model
def train_model(data):
    # Preprocess data
    X = data[['rank', 'finalWorth', 'tax_revenue_country_country', 'total_tax_rate_country', 'cpi_change_percentage', 'tax_revenue_percentage_gdp', 'total_tax_rate_percentage']]
    y = data['gender'].apply(lambda x: 1 if x == 'male' else 0)  # Convert gender to binary (0 for female, 1 for male)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Model Accuracy:", accuracy)

    return clf

# Function to perform prediction
def perform_prediction(model, input_data):
    # Perform prediction
    result = model.predict(input_data)

    # Display prediction result
    st.write("Predicted Gender:")
    if result[0] == 0:
        st.write("Female")
    else:
        st.write("Male")

# Main function
def main():
    st.title("Analisis Kekayaan Bersih orang kaya")
    data = load_data()
    st.write("Dataset:")
    st.write(data.head())

    perform_eda(data)

    # Train the model
    st.title("Memperdiksi Gender mana yang Sangat Unggul Untuk Menjadi Orang Kaya Didunia")
    st.write("Training the model:")
    model = train_model(data)

    # Input form for prediction
    st.subheader("Input Data")
    rank = st.number_input("Rank", value=1)
    finalWorth = st.number_input("Final Worth", value=0)
    tax_revenue_country_country = st.number_input("Tax Revenue Country Country", value=0)
    total_tax_rate_country = st.number_input("Total Tax Rate Country", value=0)
    cpi_change_percentage = st.number_input("CPI Change Percentage", value=0)
    tax_revenue_percentage_gdp = st.number_input("Tax Revenue Percentage GDP", value=0)
    total_tax_rate_percentage = st.number_input("Total Tax Rate Percentage", value=0)

    # Perform prediction when button is clicked
    if st.button("Predict"):
        input_data = np.array([[rank, finalWorth, tax_revenue_country_country, total_tax_rate_country, cpi_change_percentage, tax_revenue_percentage_gdp, total_tax_rate_percentage]])
        perform_prediction(model, input_data)

# Run the app
if __name__ == "__main__":
    main()
