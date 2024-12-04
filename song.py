import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

df_data_types = {
    "danceability": float,
    "energy": float,
    "key": int,
    "loudness": float,
    "mode": int,
    "speechiness": float,
    "acousticness": float,
    "instrumentalness": float,
    "liveness": float,
    "valence": float,
    "tempo": float,
    "type": str,
    "id": str,
    "uri": str,
    "track_href": str,
    "analysis_url": str,
    "duration_ms": int,
    "time_signature": int,
    "genre": str,
    "song_name": str,
}

df = pd.read_csv(
    "data/spotify/genres_v2.csv",
    dtype=df_data_types,
)

df = df.dropna(subset=["song_name", "uri"])
df = df.drop_duplicates(subset=["song_name", "uri"])

cols = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "uri",
    "genre",
    "song_name",
]
filtered_df = df[cols]
filtered_df

num_cols = [i for i in filtered_df.columns if filtered_df[i].dtype != "object"]
scaler = StandardScaler()
filtered_df.loc[:, num_cols] = scaler.fit_transform(filtered_df[num_cols])



X = filtered_df.drop(
    ["uri", "genre", "song_name"], axis=1
)  # Drop non-numeric columns if any

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
filtered_df_no_pca = filtered_df.copy()
filtered_df_no_pca["cluster"] = kmeans.fit_predict(X)

# Perform PCA
n_components = 10  # Set a higher value initially
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(filtered_df[num_cols])

# Scree plot
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = explained_variance_ratio.cumsum()

# 4. Perform PCA
n_components = 4  # Adjust the number of components as needed
pca = PCA(n_components=n_components)
pca_df = pca.fit_transform(filtered_df[num_cols])

# 5. K-Means Clustering on PCA Results
n_clusters = 4  # Number of clusters (adjust as needed)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
filtered_df_pca = filtered_df.copy()
filtered_df_pca["cluster"] = kmeans.fit_predict(pca_result)

# change the mood into string class
filtered_df_pca.loc[filtered_df_pca['cluster'] == 0, 'mood'] = 'sad'
filtered_df_pca.loc[filtered_df_pca['cluster'] == 1, 'mood'] = 'neutral'
filtered_df_pca.loc[filtered_df_pca['cluster'] == 2, 'mood'] = 'angry'
filtered_df_pca.loc[filtered_df_pca['cluster'] == 3, 'mood'] = 'happy'

print(filtered_df_pca.head())

filtered_df_pca.to_csv("data/song.csv")