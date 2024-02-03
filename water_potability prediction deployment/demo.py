import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt

# Load your trained model
model = joblib.load('trained_model.sav')

# Streamlit app
def main():
    st.title("Water Potability Prediction")

    # Input form
    st.sidebar.header("User Input Features")
    ph = st.sidebar.slider("pH", 0.0, 14.0, 7.0)
    hardness = st.sidebar.slider("Hardness", 0.0, 500.0, 150.0)
    solids = st.sidebar.slider("Solids", 0.0, 20000.0, 1000.0)
    chloramines = st.sidebar.slider("Chloramines", 0.0, 10.0, 4.0)
    sulfate = st.sidebar.slider("Sulfate", 0.0, 500.0, 200.0)
    conductivity = st.sidebar.slider("Conductivity", 0.0, 2000.0, 500.0)
    organic_carbon = st.sidebar.slider("Organic Carbon", 0.0, 50.0, 10.0)
    trihalomethanes = st.sidebar.slider("Trihalomethanes", 0.0, 150.0, 50.0)
    turbidity = st.sidebar.slider("Turbidity", 0.0, 10.0, 5.0)

    # Create a DataFrame for prediction
    input_data = pd.DataFrame({
        'ph': [ph],
        'Hardness': [hardness],
        'Solids': [solids],
        'Chloramines': [chloramines],
        'Sulfate': [sulfate],
        'Conductivity': [conductivity],
        'Organic_carbon': [organic_carbon],
        'Trihalomethanes': [trihalomethanes],
        'Turbidity': [turbidity]
    })

    # Make prediction
    prediction = model.predict(input_data)

    # Display prediction
    st.subheader("Prediction")

    # Define a message and icon based on the prediction
    if prediction[0] == 1:
        result_message = "The water is Potable"
        result_icon = "✅"
    else:
        result_message = "The water is Not Potable"
        result_icon = "❌"

    # Display the message and icon
    st.write(f"{result_icon} {result_message}")

    # Visualize feature importances
    st.subheader("Feature Importances")
    visualize_feature_importances(input_data)

def visualize_feature_importances(input_data):
    # Get feature importances
    importances = model.feature_importances_
    feature_names = input_data.columns

    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importances')
    
    # Display the plot in Streamlit
    st.pyplot()

if __name__ == "__main__":
    main()
