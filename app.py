import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load the trained model and preprocessor
with open("linear_model.pkl", "rb") as file:
    model, preprocessor = pickle.load(file)

# Set up Streamlit layout
st.set_page_config(layout="wide")  # Wide layout

# Title of the Web App
st.title("🚗 Car Price Prediction App")
st.write("This app predicts car prices based on user inputs using a trained machine learning model.")

# Initialize session state for storing prediction history
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []

# Create a sidebar for input fields
with st.sidebar:
    st.header("🔍 User Input")

    # Dropdown for Car Brand
    brand = st.selectbox("Select Car Brand:", ["Toyota", "Honda", "Ford", "BMW", "Mercedes", "Nissan", "Hyundai", "Kia", "Chevrolet", "Mazda"])

    # Dropdown for Car Model (example models for each brand)
    model_options = {
        "Toyota": ["Corolla", "Camry", "Rav4", "Prius"],
        "Honda": ["Civic", "Accord", "CR-V", "Fit"],
        "Ford": ["Focus", "Mustang", "Escape", "Explorer"],
        "BMW": ["3 Series", "5 Series", "X3", "X5"],
        "Mercedes": ["C-Class", "E-Class", "GLC", "GLE"],
        "Nissan": ["Altima", "Sentra", "Rogue", "Maxima"],
        "Hyundai": ["Elantra", "Sonata", "Tucson", "Santa Fe"],
        "Kia": ["Forte", "Optima", "Sportage", "Sorento"],
        "Chevrolet": ["Malibu", "Impala", "Equinox", "Tahoe"],
        "Mazda": ["Mazda3", "Mazda6", "CX-5", "CX-9"]
    }

    model_name = st.selectbox("Select Car Model:", model_options.get(brand, []))

    year = st.number_input("Enter Manufacturing Year:", min_value=1980, max_value=2025, value=2018)
    engine_size = st.number_input("Enter Engine Size (in liters):", min_value=0.5, max_value=10.0, value=1.8, step=0.1)
    fuel_type = st.selectbox("Select Fuel Type:", ["Petrol", "Diesel", "Electric", "Hybrid"])
    transmission = st.selectbox("Select Transmission Type:", ["Manual", "Automatic"])
    mileage = st.number_input("Enter Mileage (in km):", min_value=0, value=50000)
    doors = st.number_input("Enter Number of Doors:", min_value=2, max_value=6, value=4)
    owner_count = st.number_input("Enter Number of Previous Owners:", min_value=0, max_value=10, value=1)

# Main Content (Predict Price Button)
if st.button("Predict Price", key="predict_button"):
    # Create DataFrame for input
    new_data = pd.DataFrame([{
        "brand": brand,
        "model": model_name,
        "year": year,
        "engine_size": engine_size,
        "fuel_type": fuel_type,
        "transmission": transmission,
        "mileage": mileage,
        "doors": doors,
        "owner_count": owner_count
    }])

    # Apply preprocessing
    new_data_transformed = preprocessor.transform(new_data)

    # Make prediction
    predicted_price = model.predict(new_data_transformed)[0]

    # Add predicted price to the new_data
    new_data["Predicted Price"] = predicted_price

    # Append new prediction to history
    st.session_state["prediction_history"].append(new_data)

    # Display predicted price
    st.subheader("💰 Predicted Car Price")
    st.success(f"${predicted_price:,.2f}")

# **Collapsible Sections Below**
if st.session_state["prediction_history"]:
    
    # **📋 Collapsible Section for Data Table**
    with st.expander("📋 Data Table"):
        st.subheader("Car Details & Predictions")
        
        # Combine all stored predictions into a single DataFrame
        history_df = pd.concat(st.session_state["prediction_history"], ignore_index=True)
        
        st.table(history_df)

    # **📊 Collapsible Section for Visualization**
    with st.expander("📊 Data Visualization"):
        st.subheader("Estimated Price Trend Over Years")
        
        # Create visualization
        fig, ax = plt.subplots()
        years = list(range(2000, 2026, 2))  # Dummy years
        price_trends = [
            history_df["Predicted Price"].iloc[-1] * (1 - 0.02 * (2025 - y)) for y in years
        ]  # Example price trend

        ax.plot(years, price_trends, marker="o", linestyle="-", color="b")
        ax.set_xlabel("Year")
        ax.set_ylabel("Predicted Price ($)")
        ax.set_title("Estimated Price Trend Over Years")
        st.pyplot(fig)
