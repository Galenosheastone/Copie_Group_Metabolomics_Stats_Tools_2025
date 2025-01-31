#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:19:17 2025

@author: galen2
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from adjustText import adjust_text
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse

# Set Streamlit app title
st.title("PCA Analysis and Visualization App")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")

if uploaded_file:
    # Load dataset
    data = pd.read_csv(uploaded_file)

    # Detect Sample ID and Class/Group columns dynamically
    sample_col = data.columns[0]
    group_col = data.columns[1]

    # Prepare data for PCA
    X = data.drop([sample_col, group_col], axis=1)
    groups = data[group_col].unique()

    # Sidebar - PCA settings
    n_components = st.sidebar.slider("Number of PCA Components", 2, 3, 3)
    top_n_metabolites = st.sidebar.slider("Top Metabolites for Biplot", 5, 30, 15)

    # Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    explained_var = pca.explained_variance_ratio_ * 100
    
    # PCA DataFrame
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    pca_df['Group'] = data[group_col]
    
    # Custom colors for groups
    group_colors = sns.color_palette("husl", len(groups))
    group_color_map = {group: group_colors[i] for i, group in enumerate(groups)}
    
    # **2D PCA Plot**
    st.subheader("2D PCA Plot")
    fig, ax = plt.subplots(figsize=(8, 6))
    for group, color in group_color_map.items():
        subset = pca_df[pca_df['Group'] == group]
        ax.scatter(subset['PC1'], subset['PC2'], color=[color], label=group, alpha=0.7)
    
    ax.set_xlabel(f"PC1 ({explained_var[0]:.2f}% Variance)")
    ax.set_ylabel(f"PC2 ({explained_var[1]:.2f}% Variance)")
    ax.legend(title="Group")
    ax.grid(True)
    st.pyplot(fig)
    
    # **3D PCA Plot**
    if n_components == 3:
        st.subheader("3D PCA Plot")
        fig = go.Figure()
        for group, color in group_color_map.items():
            subset = pca_df[pca_df['Group'] == group]
            fig.add_trace(go.Scatter3d(
                x=subset['PC1'], y=subset['PC2'], z=subset['PC3'],
                mode='markers',
                marker=dict(size=6, color=f'rgb{tuple(int(c*255) for c in color)}', opacity=0.7),
                name=group
            ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title=f"PC1 ({explained_var[0]:.2f}% Variance)",
                yaxis_title=f"PC2 ({explained_var[1]:.2f}% Variance)",
                zaxis_title=f"PC3 ({explained_var[2]:.2f}% Variance)"
            ),
            title="3D PCA Plot",
            width=800, height=600
        )
        st.plotly_chart(fig)
    
    # **Loadings Visualization for First Three Components**
    st.subheader("Loadings Visualization for First Three Principal Components")
    fig, ax = plt.subplots(figsize=(8, 6))
    loadings = pca.components_.T
    for i in range(3):
        ax.plot(X.columns, loadings[:, i], label=f'PC{i+1}')
    ax.set_xlabel("Metabolites")
    ax.set_ylabel("Loadings")
    ax.legend()
    plt.xticks(rotation=90)
    st.pyplot(fig)
    
    # **2D Biplot**
    st.subheader("2D PCA Biplot")
    top_indices = np.argsort(np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2))[-top_n_metabolites:]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    for group, color in group_color_map.items():
        subset = pca_df[pca_df['Group'] == group]
        ax.scatter(subset['PC1'], subset['PC2'], color=[color], label=group, alpha=0.7)
    
    for i in top_indices:
        ax.arrow(0, 0, loadings[i, 0] * 3, loadings[i, 1] * 3, color='r', alpha=0.5)
        ax.text(loadings[i, 0] * 3.5, loadings[i, 1] * 3.5, X.columns[i], color='red', fontsize=9)
    
    ax.set_xlabel(f"PC1 ({explained_var[0]:.2f}% Variance)")
    ax.set_ylabel(f"PC2 ({explained_var[1]:.2f}% Variance)")
    ax.legend(title="Group")
    ax.grid(True)
    st.pyplot(fig)
    
    # **Interactive 3D Biplot with Labels**
    st.subheader("Interactive 3D Biplot")
    top_indices = np.argsort(np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2 + loadings[:, 2]**2))[-top_n_metabolites:]
    
    fig = go.Figure()
    for group, color in group_color_map.items():
        subset = pca_df[pca_df['Group'] == group]
        fig.add_trace(go.Scatter3d(
            x=subset['PC1'], y=subset['PC2'], z=subset['PC3'],
            mode='markers',
            marker=dict(size=6, color=f'rgb{tuple(int(c*255) for c in color)}', opacity=0.7),
            name=group
        ))
    
    for i in top_indices:
        fig.add_trace(go.Scatter3d(
            x=[0, loadings[i, 0] * 10],
            y=[0, loadings[i, 1] * 10],
            z=[0, loadings[i, 2] * 10],
            mode='lines+text',
            line=dict(color='red', width=4),
            text=["", X.columns[i]],
            textposition="top center",
            showlegend=False
        ))
    
    fig.update_layout(
        title="Interactive 3D PCA Biplot",
        width=800, height=600
    )
    st.plotly_chart(fig)
