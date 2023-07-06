Python 3.11.1 (tags/v3.11.1:a7a450f, Dec  6 2022, 19:58:39) [MSC v.1934 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> # Import necessary libraries
... !pip install streamlit
... 
... import streamlit as st
... import numpy as np
... import pandas as pd
... import os
... import re
... import tensorflow as tf
... from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Dropout, Flatten, Dense, Input, Layer
... from tensorflow.keras.applications import VGG16
... from tensorflow.keras.models import Model, Sequential
... from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
... import matplotlib.pyplot as plt
... import seaborn as sns
... from sklearn.decomposition import PCA
... from sklearn.neighbors import KNeighborsClassifier
... 
... # Set the font size for the app
... plt.rcParams['font.size'] = 16
... 
... # Set the path to the image directory
... img_path = '/kaggle/input/fashion-product-images-dataset/fashion-dataset/images'
... 
... # Load the data
... img_df = pd.read_csv('/kaggle/input/fashion-product-images-dataset/fashion-dataset/images.csv')
... styles_df = pd.read_csv('/kaggle/input/fashion-product-images-dataset/fashion-dataset/styles.csv', on_bad_lines='skip')
... 
... # Filter and sample the data
... img_files = os.listdir(img_path)
... styles_df['filename'] = styles_df['id'].astype(str) + '.jpg'
... styles_df['present'] = styles_df['filename'].apply(lambda x: x in img_files)
... styles_df = styles_df[styles_df['present']].reset_index(drop=True)
styles_df = styles_df.sample(10000)

# Set the image size
img_size = 224

# Create the image data generator
datagen = ImageDataGenerator(rescale=1/255.)
generator = datagen.flow_from_dataframe(dataframe=styles_df,
                                        directory=img_path,
                                        target_size=(img_size, img_size),
                                        x_col='filename',
                                        class_mode=None,
                                        batch_size=32,
                                        shuffle=False,
                                        classes=None)

# Load the pretrained VGG16 model
base_model = VGG16(include_top=False, input_shape=(img_size, img_size, 3))

# Freeze the pretrained layers
for layer in base_model.layers:
    layer.trainable = False

# Create the embeddings model
input_layer = Input(shape=(img_size, img_size, 3))
x = base_model(input_layer)
output = GlobalAveragePooling2D()(x)
embeddings = Model(inputs=input_layer, outputs=output)

# Generate the embeddings
X = embeddings.predict(generator, verbose=1)

# Perform dimensionality reduction with PCA
pca = PCA(2)
X_pca = pca.fit_transform(X)
styles_df[['pc1', 'pc2']] = X_pca

# Train the K-Nearest Neighbors classifier
y = styles_df['id']
nearest_neighbors = KNeighborsClassifier(n_neighbors=7)
nearest_neighbors.fit(X, y)

# Define a function to read and preprocess images
def read_img(image_path):
    image = load_img(os.path.join(img_path, image_path), target_size=(img_size, img_size, 3))
    image = img_to_array(image)
    image = image / 255.
    return image

# Create the Streamlit app
def main():
    # Set the app title and sidebar
    st.title("Fashion Similarity App")
    st.sidebar.title("Options")

    # Display the scatter plot of embeddings
    st.subheader("Visualizing Similarity")
    plt.figure(figsize=(20, 12))
    sns.scatterplot(x='pc1', y='pc2', data=styles_df)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    st.pyplot()

    # Display the similar products for a selected image
    st.subheader("Find Similar Products")
    st.write("Upload an image and find similar products.")

    # Allow the user to upload an image file
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Read and preprocess the uploaded image
        image = read_img(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Find similar products using KNN
        dist, indices = nearest_neighbors.kneighbors(X=[embeddings.predict(np.expand_dims(image, axis=0)).flatten()], n_neighbors=5)
        st.subheader("Similar Products")

        # Display the similar product images
        for i, index in enumerate(indices[0]):
            similar_image = read_img(styles_df.loc[index, 'filename'])
            st.image(similar_image, caption=f"Similar Product #{i+1}", use_column_width=True)

# Run the app
if __name__ == "__main__":
    main()
