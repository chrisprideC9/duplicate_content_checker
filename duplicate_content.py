import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import plotly.express as px

# Streamlit title and description
st.title("Duplicate Content Detection and Target Page Similarity")
st.write(
    """
    Upload a CSV file containing URLs and their vector embeddings. 
    The app identifies duplicate content and provides a target page similarity check with 3D visualization.
    """
)

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Load the CSV
    df = pd.read_csv(uploaded_file)
    
    # Clean the data to retain necessary columns
    if 'Address' not in df.columns or '(ChatGPT) Extract embeddings from page content 1' not in df.columns:
        st.error("The CSV must contain 'Address' and '(ChatGPT) Extract embeddings from page content 1' columns.")
    else:
        # Keep only the necessary columns
        df = df[['Address', '(ChatGPT) Extract embeddings from page content 1']]
        df.columns = ['url', 'vector']  # Rename columns for easier reference
        
        # Process the vector column
        try:
            df['vector'] = df['vector'].apply(lambda x: np.array(eval(x)))
        except Exception as e:
            st.error(f"Error processing vector column: {e}")
            st.stop()
        
        # Compute pairwise cosine similarity
        vectors = np.stack(df['vector'].values)
        similarity_matrix = cosine_similarity(vectors)
        
        # Compute t-SNE for visualization
        tsne = TSNE(n_components=3, random_state=42)
        vectors_3d = tsne.fit_transform(vectors)
        df['x'] = vectors_3d[:, 0]
        df['y'] = vectors_3d[:, 1]
        df['z'] = vectors_3d[:, 2]
        
        # Detected Duplicates Section
        st.subheader("Detected Duplicates")
        duplicate_threshold = st.slider(
            "Set similarity threshold for detecting duplicates", 
            min_value=0.7, max_value=1.0, value=0.9, step=0.01, key="duplicate_threshold"
        )
        
        duplicates = []
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix[i, j] >= duplicate_threshold:
                    duplicates.append((df['url'].iloc[i], df['url'].iloc[j], similarity_matrix[i, j]))
        
        if duplicates:
            duplicate_df = pd.DataFrame(duplicates, columns=["URL 1", "URL 2", "Similarity"])
            st.dataframe(duplicate_df)
        else:
            st.write("No duplicates found with the current threshold.")
        
        # Target Page Similarity Check Section
        st.subheader("Target Page Similarity Check")
        target_url = st.selectbox("Select a target URL for comparison", df['url'].tolist())
        
        if target_url:
            target_index = df[df['url'] == target_url].index[0]
            target_similarities = similarity_matrix[target_index]
            df['similarity_to_target'] = target_similarities
            
            similarity_threshold = st.slider(
                "Set similarity threshold for target page comparison", 
                min_value=0.7, max_value=1.0, value=0.9, step=0.01, key="target_threshold"
            )
            
            # Filter data for target similarity
            target_filtered = df[df['similarity_to_target'] >= similarity_threshold].copy()
            
            if not target_filtered.empty:
                st.write(f"Pages Similar to Target ({target_url}):")
                st.dataframe(target_filtered[['url', 'similarity_to_target', 'x', 'y', 'z']])
            else:
                st.write("No similar pages found with the current threshold.")
            
            # 3D Visualization for Target Similarity
            target_filtered['is_target'] = target_filtered['url'] == target_url
            fig_target = px.scatter_3d(
                target_filtered,
                x='x',
                y='y',
                z='z',
                hover_name='url',
                color='similarity_to_target',
                color_continuous_scale='Plasma',  # Different color scale for target similarity
                title="Filtered 3D Visualization of Target Page Similarity",
                width=1200,
                height=800
            )
            st.plotly_chart(fig_target)
