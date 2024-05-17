import os
import faiss
import numpy as np
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

indices_cache = {}


def update_negative_indices(displayed_images_indices, selected_images_indices):
    selected_negative_indices = [index for index in displayed_images_indices if index not in selected_images_indices]
    print("Negative indices updated:", selected_negative_indices)
    return selected_negative_indices


def find_closest_images(selected_images, features):
    selected_features = features[selected_images]
    print(selected_features)


def setup_faiss_index(features_np, index_mode="cosine"):
    dimension = features_np.shape[1]

    if index_mode == "cosine":
        index = faiss.IndexFlatIP(dimension)  # Using Inner Product for Cosine Similarity
        features_norm = features_np.copy()
        faiss.normalize_L2(features_norm)
        index.add(features_norm)
    elif index_mode == "product":
        index = faiss.IndexFlatIP(dimension)  # Using Inner Product for Cosine Similarity
        index.add(features_np)
    elif index_mode == "euclidean":
        index = faiss.IndexFlatL2(dimension)
        index.add(features_np)
    elif index_mode == "hnsw": # Hierarchical Navigable Small Worlds (HNSW)
        features_norm = features_np.copy()
        faiss.normalize_L2(features_norm)

        # Create HNSW index with inner product (IP) metric
        index = faiss.IndexHNSWFlat(dimension, 32)  # M=32 is a good default value
        index.hnsw.efConstruction = 120  # Increase for better accuracy
        index.hnsw.efSearch = 50  # Increase for better recall

        # Add normalized features to the index
        index.add(features_norm)
    elif index_mode == "ivf":
        num_clusters = 100
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, num_clusters)
        if not index.is_trained:
            index.train(features_np)
        index.add(features_np)

    return index


def refine_search(positive_indices, negative_indices, all_features, index, k=10):
    positive_indices_tensor = torch.tensor(positive_indices, dtype=torch.long)
    negative_indices_tensor = torch.tensor(negative_indices, dtype=torch.long)

    positive_features = torch.mean(all_features[positive_indices_tensor], dim=0) if positive_indices else None
    negative_features = torch.mean(all_features[negative_indices_tensor], dim=0) if negative_indices else None

    query_feature = positive_features - negative_features if negative_indices else positive_features
    query_feature = torch.nn.functional.normalize(query_feature.unsqueeze(0), p=2)

    query_feature_numpy = query_feature.numpy()
    distances, indices = index.search(query_feature_numpy, k)

    return indices.squeeze()


def compute_importance_scores(positive_indices, negative_indices, all_features):
    """
        Computes importance scores based on positive and negative indices using statistical measures.

        :param positive_indices: List of indices for positively selected images.
        :param negative_indices: List of indices for negatively selected images.
        :param all_features: Array of all features.
        :return: Importance scores for each feature dimension.
    """

    if len(positive_indices) == 0:
        return np.ones(all_features.shape[1])

    positive_features = all_features[positive_indices]
    negative_features = all_features[negative_indices] if len(negative_indices) > 0 else np.zero

    positive_mean = np.mean(positive_features, axis=0)
    negative_mean = np.mean(negative_features, axis=0)
    positive_var = np.var(positive_features, axis=0)
    negative_var = np.var(negative_features, axis=0)

    # Calculate importance scores using statistical measures
    importance_scores = np.abs(positive_mean - negative_mean) / (positive_var + negative_var + 1e-10)

    # Normalize importance scores to range [0, 1]
    importance_scores = (importance_scores - importance_scores.min()) / (
            importance_scores.max() - importance_scores.min())

    return importance_scores


def scale_features_by_importance(features, importance_scores):
    """
        Scales features by their importance scores.

        :param features: Array of features.
        :param importance_scores: Array of importance scores.
        :return: Scaled features.
        """
    return features * importance_scores


def weighted_centroid(features, sample_weights):
    """
    Calculates a weighted centroid of the features.

    :param features: Array of features.
    :param sample_weights: Array of weights for samples.
    :return: Weighted centroid.
    """
    if len(sample_weights) != len(features):
        raise ValueError("Sample weights array length must match the number of feature vectors.")

    sample_weights = sample_weights / np.sum(sample_weights)  # Normalize the weights
    weighted_features = features * sample_weights[:, np.newaxis]
    return np.sum(weighted_features, axis=0)


def refine_search_v2(positive_indices, negative_indices, all_features, index, importance_scores, weights, k=10, ):
    # Assume importance scores and weights are defined
    scaled_features = scale_features_by_importance(all_features, importance_scores)
    positive_centroid = weighted_centroid(scaled_features[positive_indices], weights[positive_indices])
    negative_centroid = weighted_centroid(scaled_features[negative_indices], weights[negative_indices])

    query_feature = positive_centroid - negative_centroid
    query_feature = faiss.normalize_L2(query_feature.reshape(1, -1))  # Normalize for cosine similarity

    # Search using the HNSW index
    _, indices = index.search(query_feature, k)
    return indices


def refine_search_v3(positive_indices, negative_indices, all_features, index, k=10):
    """
    Refines the search using positive and negative indices.

    :param positive_indices: List of positive indices.
    :param negative_indices: List of negative indices.
    :param all_features: Array of all features.
    :param index: FAISS index.
    :param k: Number of results to return.
    :return: Indices of the refined search results.
    """
    # Check if indices are valid
    if any(idx >= all_features.shape[0] for idx in positive_indices):
        raise IndexError("One of the positive indices is out of bounds.")
    if any(idx >= all_features.shape[0] for idx in negative_indices):
        raise IndexError("One of the negative indices is out of bounds.")

    importance_scores = compute_importance_scores(positive_indices, negative_indices, all_features)
    scaled_features = scale_features_by_importance(all_features, importance_scores)

    print(f"Scaled features shape: {scaled_features.shape}")
    print(f"Positive indices: {positive_indices}")
    print(f"Negative indices: {negative_indices}")

    # Use uniform weights for samples unless you have a specific method to calculate them
    positive_sample_weights = np.ones(len(positive_indices))
    negative_sample_weights = np.ones(len(negative_indices))

    positive_centroid = weighted_centroid(scaled_features[positive_indices], positive_sample_weights)
    if len(negative_indices) > 0:
        negative_centroid = weighted_centroid(scaled_features[negative_indices], negative_sample_weights)
        query_feature = positive_centroid - negative_centroid
    else:
        query_feature = positive_centroid

    # Ensure query_feature is a numpy array of type float32
    query_feature = np.array(query_feature, dtype=np.float32).reshape(1, -1)

    # Normalize query_feature using faiss.normalize_L2
    faiss.normalize_L2(query_feature)

    _, indices = index.search(query_feature, k)
    return indices.squeeze()


def iterative_search(all_features, positive_indices, negative_indices, num_iterations=5, k=10):
    """
    Performs iterative search refining based on positive and negative indices.

    :param all_features: Array of all features.
    :param positive_indices: Initial list of positive indices.
    :param negative_indices: Initial list of negative indices.
    :param num_iterations: Number of iterations for refining.
    :param k: Number of results to return each iteration.
    :return: Final list of indices after iterative search.
    """
    index = setup_faiss_index(all_features)

    for i in range(num_iterations):
        indices = refine_search_v3(positive_indices, negative_indices, all_features, index, k)

        # Update positive and negative indices for the next iteration
        positive_indices = list(set(positive_indices + list(indices)))
        negative_indices = [index for index in range(len(all_features)) if index not in positive_indices]

        print(f"Iteration {i + 1}: {len(positive_indices)} positive, {len(negative_indices)} negative")

    return indices
