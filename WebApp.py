#!/usr/bin/env python
# coding: utf-8

# In[19]:


# Importing packages
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from PIL import Image
from random import randint
from sklearn.metrics import silhouette_score

# K-Means Clustering

# Load the data from CSV file
data = pd.read_csv('product_images.csv')

# Convert the data to a numpy array
X = np.array(data)

# Set the number of clusters to use
k = 7

# Initialize the KMeans model with the chosen number of clusters
kmeans = KMeans(n_clusters = k, random_state = 42)

# Fit the KMeans model to the data
kmeans.fit(X)

# Assign each row in the data to a centroid
centroids = kmeans.predict(X)

# Create a new dataframe to store the results
result_df = pd.DataFrame(data = X)

# Add a new column to the dataframe with the assigned centroid for each row
result_df['centroid'] = centroids

silhouette_score = round(silhouette_score(X, centroids),3)

# Define a function to get the 5 closest images to a given centroid
def get_closest_images(centroid):
    # Get the indices of rows that belong to the selected centroid
    indices = np.where(result_df['centroid'] == centroid)[0]
    if len(indices) == 0:
        return np.array([])
    # Calculate the distance between each row and the centroid
    distances = np.linalg.norm(X[indices] - kmeans.cluster_centers_[centroid], axis=1)
    # Sort the distances and get the indices of the 5 closest images
    closest_indices = indices[np.argsort(distances)[:5]]
    # Return the 5 closest images
    return X[closest_indices]


# Define a function to get the next 5 closest images to a given centroid
def get_next_closest_images(centroid):
    # Get the indices of rows that belong to the selected centroid
    indices = np.where(result_df['centroid'] == centroid)[0]
    if len(indices) == 0:
        return np.array([])
    # Calculate the distance between each row and the centroid
    distances = np.linalg.norm(X[indices] - kmeans.cluster_centers_[centroid], axis=1)
    # Sort the distances and get the indices of the next 5 closest images (except for the first 5 closest ones)
    next_closest_indices = indices[np.argsort(distances)[:26]]
    # Return the next 5 closest images
    randomlist2 = []
    for i in range(0,5):
        randomlist2.append(randint(0, 25))
    return X[next_closest_indices][randomlist2]

# Defining a function for the cart
def added_to_cart():
    st.session_state.cart_counter += 1
    
if 'cart_counter' not in st.session_state:
    st.session_state.cart_counter = 0 
# Building the web app

# Defining a dictionary to map centroid numbers to names
centroid_names = {0: 'Boots', 1: 'Pants', 2: 'Outerwear', 3: 'Sneakers', 4: 'Bags', 5: 'T-shirts', 6: 'Long Sleeves'}

# Creating a dropdown menu for selecting a centroid
st.set_page_config(page_title = "Product Recommendation")
st.title("Product Recommendation")
st.write("Select a product category and we will recommend a few products to you!")
selected_centroid = st.selectbox('Product Categories:', [''] + list(centroid_names.values()))

# Get the closest images to the selected centroid if a valid centroid was selected
if selected_centroid != '':
    centroid_num = [k for k, v in centroid_names.items() if v == selected_centroid][0]
    closest_images = get_closest_images(centroid_num)
    if len(closest_images) > 0:
        # Display the closest images using Matplotlib
        fig, axs = plt.subplots(1, 5, figsize=(10, 2))
        for i, image in enumerate(closest_images):
            axs[i].imshow(np.reshape(image, (28, 28)), cmap=plt.cm.binary)
            axs[i].axis('off')

        st.pyplot(fig)


    # Add 5 buttons with unique text output for each button
    button_names = ['Product 1', 'Product 2', 'Product 3', 'Product 4', 'Product 5']

    button_col1, button_col2, button_col3, button_col4, button_col5 = st.columns(5)

    with button_col1:
        st.checkbox(button_names[0])

    with button_col2:
        st.checkbox(button_names[1])

    with button_col3:
        st.checkbox(button_names[2])

    with button_col4:
        st.checkbox(button_names[3])

    with button_col5:
        st.checkbox(button_names[4])

# Add a "Show More" button to display additional images
if selected_centroid != '':
    centroid_num = [k for k, v in centroid_names.items() if v == selected_centroid][0]
    closest_images = get_closest_images(centroid_num)
    if len(closest_images) > 0:
        # Display the closest images using Matplotlib
        fig, axs = plt.subplots(1, 5, figsize=(10, 2))
        for i, image in enumerate(closest_images):
            axs[i].imshow(np.reshape(image, (28, 28)), cmap=plt.cm.binary)
            axs[i].axis('off')

        if st.button("**Suggest Alternatives**"):
            # Get the next 5 closest images to the selected centroid
            next_closest_images = get_next_closest_images(centroid_num)
            if len(next_closest_images) > 0:
                # Display the next closest images using Matplotlib

                fig, axs = plt.subplots(1, 5, figsize=(10, 2))
                for i, image in enumerate(next_closest_images):
                    axs[i].imshow(np.reshape(image, (28, 28)), cmap=plt.cm.binary)
                    axs[i].axis('off')

                st.pyplot(fig)
    
                
        button_col6, button_col7, button_col8, button_col9, button_col10 = st.columns(5)

        with button_col6:
            if st.button("Add to Cart"):
                added_to_cart()
                st.success("Product has been added to your cart!")

        with button_col7:
            if st.button("Add to Cart "):
                added_to_cart()
                st.success("Product has been added to your cart!")

        with button_col8:
            if st.button("Add to Cart  "):
                added_to_cart()
                st.success("Product has been added to your cart!")


        with button_col9:               
            if st.button("Add to Cart   "):
                added_to_cart()
                st.success("Product has been added to your cart!")


        with button_col10:              
            if st.button("Add to Cart    "):
                added_to_cart()
                st.success("Product has been added to your cart!")

        total_count = st.session_state.cart_counter
        st.sidebar.markdown(f"🛒 **{total_count}**")    

        # S Score
        st.sidebar.write("The Silhouette Score with 7 clusters is ", silhouette_score)


        

