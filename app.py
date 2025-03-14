import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("Interactive Data Science App")

# User input for selecting dataset
dataset = st.selectbox("Select a dataset", ["Iris", "Wine", "Load Dataset"])

# Handle different datasets
if dataset == "Iris":
    # Load the Iris dataset
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target

elif dataset == "Wine":
    # Load the Wine dataset
    from sklearn.datasets import load_wine
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target

else:
    # Allow user to upload a custom dataset
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)

# Show the first few rows of the dataset
st.write("Here is a preview of the dataset:", df.head())

# Create an interactive plot
st.subheader("Interactive Scatterplot")
x_axis = st.selectbox("Select the X axis", df.columns[:-1])
y_axis = st.selectbox("Select the Y axis", df.columns[:-1])

fig, ax = plt.subplots()
sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax)
st.pyplot(fig)

# Add a button for download
st.download_button(
    label="Download the dataset as CSV",
    data=df.to_csv(index=False).encode(),
    file_name="dataset.csv",
    mime="text/csv"
)
