import numpy as np
from sklearn.cluster import KMeans

class CentroidAnonymizer:
    def __init__(self, args, device='cpu'):
        self.args = args
        self.seed = args.seed
        self.device = device
        
        self.k = args.k_same
        self.clustering = args.clustering
        self.return_centroids = args.return_centroids
        self.dp_noise = args.centroids_dp_noise
        self.epsilon = args.epsilon
        self.delta = args.delta
        self.max_feat_norm = args.max_feat_norm
    
    def apply(self, train_features, train_labels):
        """
        clustering: "unsupervised"
        return_centroids: If True, return (noisy) centroids + labels
                          If False, return anonymized data (k-Same style) + labels
        dp_noise: None, "laplacian", or "gaussian"
        """
        # Store features/labels for sensitivity calculation
        self.train_features = train_features
        self.train_labels = train_labels

        # Clip norms if applying DP
        self.train_features = self._clip_features(self.train_features) if self.dp_noise else self.train_features
    
        # Route to appropriate clustering method
        if self.clustering == "unsupervised":
            centroids, labels = self._compute_local_centroids()
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering}")
            
        # Apply DP noise if requested
        centroids = self._add_dp_noise(centroids, self.dp_noise, self.epsilon, self.delta) if self.dp_noise else centroids
        
        # Prepare return values
        if self.return_centroids:
            return centroids, labels
        else:
            anonymized_data = np.array([centroids[l] for l in self.cluster_labels])

            if self.clustering == "unsupervised":
                return anonymized_data, self.cluster_labels
            else: 
                return anonymized_data, self.train_labels 
    
    def _clip_features(self, features):
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        factors = np.minimum(1.0, self.max_feat_norm / (norms + 1e-10))
        return features * factors
    
    def _compute_local_centroids(self):
        """Unsupervised clustering using `k` for minimum group size"""
        n_samples = len(self.train_features)
        n_clusters = max(1, n_samples // self.k)

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed)
        self.cluster_labels = kmeans.fit_predict(self.train_features)
        centroids = kmeans.cluster_centers_
        labels = [
            np.argmax(np.bincount(self.train_labels[self.cluster_labels == i]))
            for i in range(n_clusters)
        ]
        return centroids, labels
    
    def _add_dp_noise(self, centroids, noise_type, epsilon, delta=None, max_feat_norm=1.5):
        """Add DP noise with post-noise normalization"""
        
        # USE GLOBAL DP SENSITIVITY (stricter privacy)
        k, d = centroids.shape
        noisy_centroids = np.zeros_like(centroids)

        for i in range(k):
            # Count of points in this cluster
            cluster_points = self.train_features[self.cluster_labels == i]
            n_i = len(cluster_points)

            if n_i == 0:
                noisy_centroids[i] = centroids[i]
                continue

            # Global sensitivity for the mean
            sensitivity = 2 * max_feat_norm / n_i

            if noise_type == "laplacian":
                scale = sensitivity / epsilon
                noise = np.random.laplace(0, scale, size=(d,))
            elif noise_type == "gaussian":
                if delta is None:
                    raise ValueError("delta must be specified for Gaussian noise")
                sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
                noise = np.random.normal(0, sigma, size=(d,))
            else:
                raise ValueError("Invalid noise_type")

            noisy_centroids[i] = centroids[i] + noise

        return noisy_centroids


        