import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class ColdStartHandler:
    """
    Comprehensive cold start solution for recommendation systems.
    Handles both cold start users and cold start items.
    """
    
    def __init__(self):
        self.user_clusters = None
        self.item_clusters = None
        self.popular_items = []
        self.diverse_items = []
        self.trending_items = []
        
    def fit(self, user_item_matrix, user_mapping, item_mapping, n_clusters=5):
        """
        Fit the cold start handler on training data.
        
        Args:
            user_item_matrix (csr_matrix): User-item interaction matrix
            user_mapping (dict): User ID to index mapping
            item_mapping (dict): Item ID to index mapping
            n_clusters (int): Number of clusters for user/item clustering
        """
        logger.info("Fitting cold start handler...")
        
        # Create user profiles for clustering
        self._create_user_clusters(user_item_matrix, n_clusters)
        
        # Create item profiles for clustering
        self._create_item_clusters(user_item_matrix, n_clusters)
        
        # Identify popular items
        self._identify_popular_items(user_item_matrix, item_mapping)
        
        # Identify diverse items
        self._identify_diverse_items(user_item_matrix, item_mapping)
        
        # Identify trending items (proxy based on user activity)
        self._identify_trending_items(user_item_matrix, item_mapping)
        
        logger.info("Cold start handler fitted successfully")
    
    def _create_user_clusters(self, user_item_matrix, n_clusters):
        """Create user clusters based on interaction patterns."""
        # Create user feature vectors
        user_features = []
        
        for user_idx in range(user_item_matrix.shape[0]):
            user_row = user_item_matrix[user_idx]
            
            # Basic features
            total_interactions = user_row.sum()
            unique_items = user_row.nnz
            avg_rating = total_interactions / unique_items if unique_items > 0 else 0
            
            # Category preferences (simulate by item index ranges)
            n_items = user_item_matrix.shape[1]
            category_prefs = []
            for i in range(5):  # 5 simulated categories
                start_idx = int(i * n_items / 5)
                end_idx = int((i + 1) * n_items / 5)
                category_interactions = user_row[:, start_idx:end_idx].sum()
                category_prefs.append(float(category_interactions))
            
            features = [float(total_interactions), float(unique_items), float(avg_rating)] + category_prefs
            user_features.append(features)
        
        # Cluster users
        if len(user_features) > n_clusters:
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(user_features)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            self.user_clusters = kmeans.fit_predict(normalized_features)
        else:
            self.user_clusters = np.zeros(len(user_features))
    
    def _create_item_clusters(self, user_item_matrix, n_clusters):
        """Create item clusters based on user interaction patterns."""
        # Create item feature vectors
        item_features = []
        
        for item_idx in range(user_item_matrix.shape[1]):
            item_col = user_item_matrix[:, item_idx]
            
            # Basic features
            total_interactions = item_col.sum()
            unique_users = item_col.nnz
            avg_rating = total_interactions / unique_users if unique_users > 0 else 0
            
            # User segment preferences (simulate by user index ranges)
            n_users = user_item_matrix.shape[0]
            segment_prefs = []
            for i in range(5):  # 5 simulated segments
                start_idx = int(i * n_users / 5)
                end_idx = int((i + 1) * n_users / 5)
                segment_interactions = item_col[start_idx:end_idx].sum()
                segment_prefs.append(float(segment_interactions))
            
            features = [float(total_interactions), float(unique_users), float(avg_rating)] + segment_prefs
            item_features.append(features)
        
        # Cluster items
        if len(item_features) > n_clusters:
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(item_features)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            self.item_clusters = kmeans.fit_predict(normalized_features)
        else:
            self.item_clusters = np.zeros(len(item_features))
    
    def _identify_popular_items(self, user_item_matrix, item_mapping, top_k=50):
        """Identify most popular items."""
        item_popularity = np.array(user_item_matrix.sum(axis=0)).flatten()
        top_indices = np.argsort(item_popularity)[::-1][:top_k]
        
        reverse_item_mapping = {idx: item for item, idx in item_mapping.items()}
        self.popular_items = [
            (reverse_item_mapping[idx], float(item_popularity[idx]))
            for idx in top_indices if idx in reverse_item_mapping
        ]
    
    def _identify_diverse_items(self, user_item_matrix, item_mapping, top_k=50):
        """Identify diverse items across different user segments."""
        if self.item_clusters is None:
            return
        
        diverse_items = []
        unique_clusters = np.unique(self.item_clusters)
        
        reverse_item_mapping = {idx: item for item, idx in item_mapping.items()}
        
        for cluster in unique_clusters:
            cluster_items = np.where(self.item_clusters == cluster)[0]
            if len(cluster_items) > 0:
                # Get most popular item from each cluster
                cluster_popularity = np.array(user_item_matrix[:, cluster_items].sum(axis=0)).flatten()
                best_item_idx = cluster_items[np.argmax(cluster_popularity)]
                
                if best_item_idx in reverse_item_mapping:
                    item_id = reverse_item_mapping[best_item_idx]
                    popularity = float(user_item_matrix[:, best_item_idx].sum())
                    diverse_items.append((item_id, popularity))
        
        # Sort by popularity and take top items
        diverse_items.sort(key=lambda x: x[1], reverse=True)
        self.diverse_items = diverse_items[:top_k]
    
    def _identify_trending_items(self, user_item_matrix, item_mapping, top_k=30):
        """Identify trending items based on recent user activity."""
        # Use last 30% of users as proxy for recent activity
        n_users = user_item_matrix.shape[0]
        recent_start = int(n_users * 0.7)
        recent_matrix = user_item_matrix[recent_start:, :]
        
        recent_popularity = np.array(recent_matrix.sum(axis=0)).flatten()
        top_indices = np.argsort(recent_popularity)[::-1][:top_k]
        
        reverse_item_mapping = {idx: item for item, idx in item_mapping.items()}
        self.trending_items = [
            (reverse_item_mapping[idx], float(recent_popularity[idx]))
            for idx in top_indices if idx in reverse_item_mapping
        ]
    
    def recommend_for_cold_start_user(self, n_recommendations=10, strategy='hybrid'):
        """
        Generate recommendations for completely new users.
        
        Args:
            n_recommendations (int): Number of recommendations
            strategy (str): Recommendation strategy ('popular', 'diverse', 'trending', 'hybrid')
            
        Returns:
            list: List of (item_id, score, explanation) tuples
        """
        recommendations = []
        
        if strategy == 'popular':
            items = self.popular_items[:n_recommendations]
            for item_id, score in items:
                explanation = f"Popular item ({int(score)} total interactions)"
                recommendations.append((item_id, score / max(s for _, s in self.popular_items), explanation))
                
        elif strategy == 'diverse':
            items = self.diverse_items[:n_recommendations]
            for item_id, score in items:
                explanation = f"Diverse recommendation from different category"
                recommendations.append((item_id, score / max(s for _, s in self.diverse_items), explanation))
                
        elif strategy == 'trending':
            items = self.trending_items[:n_recommendations]
            for item_id, score in items:
                explanation = f"Trending item ({int(score)} recent interactions)"
                recommendations.append((item_id, score / max(s for _, s in self.trending_items), explanation))
                
        elif strategy == 'hybrid':
            # Combine all strategies
            popular_weight = 0.5
            diverse_weight = 0.3
            trending_weight = 0.2
            
            item_scores = {}
            
            # Add popular items
            max_popular = max(s for _, s in self.popular_items) if self.popular_items else 1
            for item_id, score in self.popular_items:
                item_scores[item_id] = item_scores.get(item_id, 0) + popular_weight * (score / max_popular)
            
            # Add diverse items
            max_diverse = max(s for _, s in self.diverse_items) if self.diverse_items else 1
            for item_id, score in self.diverse_items:
                item_scores[item_id] = item_scores.get(item_id, 0) + diverse_weight * (score / max_diverse)
            
            # Add trending items
            max_trending = max(s for _, s in self.trending_items) if self.trending_items else 1
            for item_id, score in self.trending_items:
                item_scores[item_id] = item_scores.get(item_id, 0) + trending_weight * (score / max_trending)
            
            # Sort and select top items
            sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
            
            for item_id, score in sorted_items:
                explanation = "Hybrid cold start recommendation (popular + diverse + trending)"
                recommendations.append((item_id, float(score), explanation))
        
        return recommendations
    
    def recommend_similar_to_cluster(self, user_cluster, n_recommendations=10):
        """
        Recommend items based on user cluster preferences.
        
        Args:
            user_cluster (int): User cluster ID
            n_recommendations (int): Number of recommendations
            
        Returns:
            list: List of (item_id, score, explanation) tuples
        """
        if self.item_clusters is None:
            return self.recommend_for_cold_start_user(n_recommendations, 'popular')
        
        # Find items preferred by similar users
        cluster_items = []
        for item_idx, item_cluster in enumerate(self.item_clusters):
            if item_cluster == user_cluster:
                cluster_items.append(item_idx)
        
        if not cluster_items:
            return self.recommend_for_cold_start_user(n_recommendations, 'diverse')
        
        # For now, return popular items (can be enhanced with actual cluster analysis)
        return self.recommend_for_cold_start_user(n_recommendations, 'hybrid')
    
    def get_cold_start_statistics(self):
        """Get statistics about cold start recommendations."""
        return {
            'popular_items_count': len(self.popular_items),
            'diverse_items_count': len(self.diverse_items),
            'trending_items_count': len(self.trending_items),
            'user_clusters': len(np.unique(self.user_clusters)) if self.user_clusters is not None else 0,
            'item_clusters': len(np.unique(self.item_clusters)) if self.item_clusters is not None else 0
        }