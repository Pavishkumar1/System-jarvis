import numpy as np
import pandas as pd
import streamlit as st


# Function to load data from uploaded CSV file
def load_data(file):
    data = pd.read_csv(file)
    return data

# Function to learn using Candidate Elimination Algorithm
def learn(concepts, target):
        # Initialise S0 with the first instance from concepts
    specific_h = concepts[0].copy()

    general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]

    # The learning iterations
    for i, h in enumerate(concepts):

        # Checking if the hypothesis has a positive target
        if target[i] == "Yes":
            for x in range(len(specific_h)):
                # Change values in S & G only if values change
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'

        # Checking if the hypothesis has a negative target
        if target[i] == "No":
            for x in range(len(specific_h)):
                # For negative hypothesis change values only in G
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

    # Find indices where we have empty rows, meaning those that are unchanged
    indices = [i for i, val in enumerate(general_h) if val == ['?'] * len(specific_h)]
    for i in indices:
        # Remove those rows from general_h
        general_h.remove(['?'] * len(specific_h))
    # Return final values
    return specific_h, general_h



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
