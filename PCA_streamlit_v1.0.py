#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 17:03:53 2025

@author: galen2
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_data(data):
    try:
        data = data.dropna()
        features = data.iloc[:, 2:]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        data.iloc[:, 2:] = scaled_features
        return data
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return None

def perform_pca(data):
    try:
        features = data.iloc[:, 2:]
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(features)
        return principal_components, pca.explained_variance_ratio_
    except Exception as e:
        st.error(f"Error performing PCA: {e}")
        return None, None

def plot_pca(principal_components, labels):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=principal_components[:, 0], y=principal_components[:, 1], hue=labels, palette='viridis')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Analysis")
    plt.legend(title="Groups")
    plt.grid(True)
    st.pyplot(plt)

st.title("PCA Analysis Tool")
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    if data is not None:
        st.write("### Raw Data")
        st.dataframe(data.head())
        processed_data = preprocess_data(data)
        if processed_data is not None:
            st.write("### Processed Data")
            st.dataframe(processed_data.head())
            principal_components, explained_variance = perform_pca(processed_data)
            if principal_components is not None:
                st.write(f"### Explained Variance Ratio: {explained_variance}")
                st.write("### PCA Plot")
                plot_pca(principal_components, processed_data.iloc[:, 1])
