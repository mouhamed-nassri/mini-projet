import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD

# -----------------------
# DATASET SIMPLIFIÉ
# -----------------------
ratings = np.array([
    [5, 0, 3, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [0, 2, 4, 4]
])

# -----------------------
# KNN MODEL
# -----------------------
knn = NearestNeighbors(metric="cosine")
knn.fit(ratings)

# -----------------------
# SVD MODEL
# -----------------------
svd = TruncatedSVD(n_components=2)
svd_matrix = svd.fit_transform(ratings)

# -----------------------
# RECOMMANDATION
# -----------------------
def recommend_for_user(user_id: int):
    distances, indices = knn.kneighbors([ratings[user_id]], n_neighbors=2)
    return indices.flatten().tolist()

# -----------------------
# COMPARAISON MODELES
# -----------------------
def compare_models():
    return {
        "KNN": {
            "type": "similarity-based",
            "speed": "fast",
            "accuracy": "medium"
        },
        "SVD": {
            "type": "matrix-factorization",
            "speed": "medium",
            "accuracy": "high"
        }
    }