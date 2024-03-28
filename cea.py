import numpy as np
import pandas as pd
import streamlit as st

# Function to load data from uploaded CSV file
def load_data(file):
    data = pd.read_csv(file)
    return data

# Function to learn using Candidate Elimination Algorithm
def learn(concepts, target):
    # Implementation of the learning method
    # Your existing learn function code goes here

# Streamlit app
st.title('Candidate-Elimination Algorithm')

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    # Load data from uploaded file
    data = load_data(uploaded_file)

    # Separating concept features from Target
    concepts = np.array(data.iloc[:, 0:-1])
    target = np.array(data.iloc[:, -1])

    # Execute the algorithm
    s_final, g_final = learn(concepts, target)

    # Display the training data
    st.subheader('Training Data')
    st.write(data)

    # Display final specific hypothesis
    st.subheader('Final Specific Hypothesis')
    st.write(s_final)

    # Convert final general hypotheses to DataFrame for tabular display
    g_final_df = pd.DataFrame(g_final, columns=data.columns[:-1])
    
    # Display final general hypotheses
    st.subheader('Final General Hypotheses')
    st.write(g_final_df)
