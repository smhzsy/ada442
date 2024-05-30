import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
loaded_model = pickle.load(open('gbc_model.pkl', 'rb'))

# Title and description
st.title('Bank Marketing Prediction')
st.markdown("""
    This application predicts whether a customer will respond positively to a bank marketing campaign based on their details.
    Please enter the customer details below and click 'Predict'.
""")

# Sidebar for user input
st.sidebar.header('Enter Customer Details')


def user_input_features():
    age = st.sidebar.slider('Age', 18, 100, 35)
    job = st.sidebar.selectbox('Job', ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'], index=4)
    marital = st.sidebar.selectbox('Marital Status', ['married', 'single', 'divorced', 'unknown'], index=0)
    education = st.sidebar.selectbox('Education', ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown'], index=6)
    default = st.sidebar.selectbox('Credit in Default', ['no', 'yes', 'unknown'], index=0)
    housing = st.sidebar.selectbox('Housing Loan', ['no', 'yes', 'unknown'], index=1)
    loan = st.sidebar.selectbox('Personal Loan', ['no', 'yes', 'unknown'], index=0)
    contact = st.sidebar.selectbox('Contact Communication', ['cellular', 'telephone'], index=0)
    month = st.sidebar.selectbox('Last Contact Month', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], index=4)
    day_of_week = st.sidebar.selectbox('Last Contact Day of the Week', ['mon', 'tue', 'wed', 'thu', 'fri'], index=4)
    duration = st.sidebar.number_input('Last Contact Duration (seconds)', min_value=0, max_value=5000, value=300)
    campaign = st.sidebar.number_input('Number of Contacts During Campaign', min_value=1, max_value=50, value=2)
    pdays = st.sidebar.number_input('Number of Days Since Last Contact', min_value=0, max_value=999, value=999)
    previous = st.sidebar.number_input('Number of Contacts Before Campaign', min_value=0, max_value=50, value=1)
    poutcome = st.sidebar.selectbox('Outcome of Previous Campaign', ['failure', 'nonexistent', 'success'], index=2)
    emp_var_rate = st.sidebar.number_input('Employment Variation Rate', value=-1.8, step=0.1)
    cons_price_idx = st.sidebar.number_input('Consumer Price Index', value=92.893, step=0.001)
    cons_conf_idx = st.sidebar.number_input('Consumer Confidence Index', value=-46.2, step=0.1)
    euribor3m = st.sidebar.number_input('Euribor 3 Month Rate', value=1.313, step=0.001)
    nr_employed = st.sidebar.number_input('Number of Employees', value=5000, step=0.1)

    data = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'day_of_week': day_of_week,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome,
        'emp.var.rate': emp_var_rate,
        'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m,
        'nr.employed': nr_employed,
    }
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()

# Display user input
st.write("## Customer Details")
st.write(input_df)

# Predict button
if st.button('Predict'):
    prediction = loaded_model.predict(input_df)
    prediction_prob = loaded_model.predict_proba(input_df)

    st.write("### Prediction Result")
    st.write(f"Prediction: {'Yes' if prediction[0] == 'yes' else 'No'}")
    st.write(f"Prediction Probability: {prediction_prob[0][1]:.2f}")

    if prediction[0] == 'yes':
        st.success('The customer is likely to respond positively to the campaign.')
    else:
        st.error('The customer is unlikely to respond positively to the campaign.')

# Footer
st.markdown("""
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            text-align: center;
            padding: 10px;
        }
    </style>
    <div class="footer">
        <p>Created by Your Name - 2024</p>
    </div>
""", unsafe_allow_html=True)