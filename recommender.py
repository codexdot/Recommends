import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImplicitRecommender:
    """
    Matrix factorization-based recommendation system for implicit feedback.
    Uses Non-negative Matrix Factorization (NMF) as the core algorithm.
    """
    
    def __init__(self, algorithm='nmf', factors=50, regularization=0.1, 
                 iterations=20, alpha=15, random_state=42):
        """
        Initialize the recommender.
        
        Args:
            algorithm (str): Algorithm to use ('nmf', 'als', 'bpr', 'lmf')
            factors (int): Number of latent factors
            regularization (float): Regularization parameter
            iterations (int): Number of training iterations
            alpha (float): Confidence parameter for implicit feedback
            random_state (int): Random seed
        """
        self.algorithm = algorithm
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state
        
        # Initialize NMF model (using sklearn's implementation)
        self.model = NMF(
            n_components=factors,
            init='random',
            random_state=random_state,
            max_iter=iterations,
            alpha_W=regularization,
            alpha_H=regularization
        )
        
        self.is_fitted = False
        self.user_factors = None
        self.item_factors = None
        logger.info(f"Initialized {algorithm} recommender with {factors} factors")
    
    def fit(self, user_item_matrix):
        """
        Train the recommendation model.
        
        Args:
            user_item_matrix (csr_matrix): User-item interaction matrix
        """
        logger.info("Training recommendation model...")
        
        # Convert to dense matrix for NMF (it requires non-negative values)
        dense_matrix = user_item_matrix.toarray()
        
        # Apply confidence weighting for implicit feedback
        confidence_matrix = 1 + self.alpha * dense_matrix
        
        # Fit the NMF model
        self.user_factors = self.model.fit_transform(confidence_matrix)
        self.item_factors = self.model.components_.T
        
        self.is_fitted = True
        self.user_item_matrix = user_item_matrix
        
        logger.info("Model training completed")
    
    def recommend_for_user(self, user_id, user_item_matrix, user_mapping, item_mapping, 
                          n_recommendations=10, include_explanations=False):
        """
        Generate recommendations for a specific user.
        
        Args:
            user_id: Original user ID
            user_item_matrix (csr_matrix): User-item interaction matrix
            user_mapping (dict): User ID to index mapping
            item_mapping (dict): Item ID to index mapping
            n_recommendations (int): Number of recommendations to generate
            include_explanations (bool): Whether to include explanations
            
        Returns:
            list: List of (item_id, score, explanation) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Handle cold start users
        if user_id not in user_mapping:
            logger.warning(f"User {user_id} not in training data. Using cold start recommendations.")
            return self._get_cold_start_recommendations(
                user_item_matrix, item_mapping, n_recommendations, include_explanations
            )
        
        user_idx = user_mapping[user_id]
        
        # Get recommendations from the model
        item_user_matrix = user_item_matrix.T.tocsr()
        
        try:
            recommended_items, scores = self.model.recommend(
                user_idx, 
                user_item_matrix[user_idx],
                N=n_recommendations,
                filter_already_liked_items=True
            )
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return self._get_cold_start_recommendations(
                user_item_matrix, item_mapping, n_recommendations, include_explanations
            )
        
        # Map back to original item IDs
        reverse_item_mapping = {idx: item for item, idx in item_mapping.items()}
        
        recommendations = []
        for item_idx, score in zip(recommended_items, scores):
            item_id = reverse_item_mapping[item_idx]
            
            explanation = None
            if include_explanations:
                explanation = self._generate_explanation(
                    user_id, item_id, user_item_matrix, user_mapping, item_mapping
                )
            
            recommendations.append((item_id, float(score), explanation))
        
        return recommendations
    
    def _get_cold_start_recommendations(self, user_item_matrix, item_mapping, 
                                      n_recommendations, include_explanations):
        """
        Generate recommendations for cold start users based on item popularity.
        
        Args:
            user_item_matrix (csr_matrix): User-item interaction matrix
            item_mapping (dict): Item ID to index mapping
            n_recommendations (int): Number of recommendations
            include_explanations (bool): Whether to include explanations
            
        Returns:
            list: List of (item_id, score, explanation) tuples
        """
        # Calculate item popularity
        item_popularity = np.array(user_item_matrix.sum(axis=0)).flatten()
        
        # Get top popular items
        top_item_indices = np.argsort(item_popularity)[::-1][:n_recommendations]
        
        reverse_item_mapping = {idx: item for item, idx in item_mapping.items()}
        
        recommendations = []
        for item_idx in top_item_indices:
            item_id = reverse_item_mapping[item_idx]
            popularity_score = item_popularity[item_idx]
            
            # Normalize popularity score to [0, 1]
            max_popularity = np.max(item_popularity)
            normalized_score = popularity_score / max_popularity if max_popularity > 0 else 0
            
            explanation = None
            if include_explanations:
                explanation = f"Popular item (interacted with by {int(popularity_score)} users)"
            
            recommendations.append((item_id, float(normalized_score), explanation))
        
        return recommendations
    
    def _generate_explanation(self, user_id, item_id, user_item_matrix, 
                            user_mapping, item_mapping):
        """
        Generate simple explanation for a recommendation.
        
        Args:
            user_id: Original user ID
            item_id: Original item ID
            user_item_matrix (csr_matrix): User-item interaction matrix
            user_mapping (dict): User ID to index mapping
            item_mapping (dict): Item ID to index mapping
            
        Returns:
            str: Explanation text
        """
        try:
            # Get similar items based on user's interaction history
            user_idx = user_mapping[user_id]
            item_idx = item_mapping[item_id]
            
            # Get user's interacted items
            user_items = user_item_matrix[user_idx].nonzero()[1]
            
            if len(user_items) == 0:
                return "Recommended based on overall popularity"
            
            # Find similar users who liked this item
            item_users = user_item_matrix[:, item_idx].nonzero()[0]
            
            # Count overlap with user's items
            similar_interactions = 0
            for other_user_idx in item_users:
                other_user_items = set(user_item_matrix[other_user_idx].nonzero()[1])
                user_items_set = set(user_items)
                overlap = len(user_items_set.intersection(other_user_items))
                if overlap > 0:
                    similar_interactions += overlap
            
            if similar_interactions > 0:
                return f"Users with similar preferences also liked this item"
            else:
                return "Recommended based on collaborative filtering"
                
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return "Recommended by the system"
    
    def get_similar_items(self, item_id, item_mapping, n_similar=10):
        """
        Get items similar to a given item.
        
        Args:
            item_id: Original item ID
            item_mapping (dict): Item ID to index mapping
            n_similar (int): Number of similar items to return
            
        Returns:
            list: List of (item_id, similarity_score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before finding similar items")
        
        if item_id not in item_mapping:
            return []
        
        item_idx = item_mapping[item_id]
        
        try:
            similar_items, similarities = self.model.similar_items(
                item_idx, N=n_similar + 1  # +1 because it includes the item itself
            )
            
            # Remove the item itself from results
            similar_items = similar_items[1:]
            similarities = similarities[1:]
            
            reverse_item_mapping = {idx: item for item, idx in item_mapping.items()}
            
            return [(reverse_item_mapping[idx], float(sim)) 
                   for idx, sim in zip(similar_items, similarities)]
        
        except Exception as e:
            logger.error(f"Error finding similar items: {e}")
            return []
    
    def get_model_size(self):
        """
        Estimate model size in MB.
        
        Returns:
            float: Model size in MB
        """
        if not self.is_fitted:
            return 0.0
        
        # Estimate size based on factor matrices
        try:
            user_factors = self.model.user_factors
            item_factors = self.model.item_factors
            
            size_bytes = user_factors.nbytes + item_factors.nbytes
            size_mb = size_bytes / (1024 ** 2)
            
            return size_mb
        except Exception:
            # Fallback estimation
            return self.factors * 2 * 8 / (1024 ** 2)  # Rough estimate
    
    def get_user_embedding(self, user_id, user_mapping):
        """
        Get user embedding vector.
        
        Args:
            user_id: Original user ID
            user_mapping (dict): User ID to index mapping
            
        Returns:
            np.ndarray: User embedding vector
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if user_id not in user_mapping:
            return None
        
        user_idx = user_mapping[user_id]
        return self.model.user_factors[user_idx]
    
    def get_item_embedding(self, item_id, item_mapping):
        """
        Get item embedding vector.
        
        Args:
            item_id: Original item ID
            item_mapping (dict): Item ID to index mapping
            
        Returns:
            np.ndarray: Item embedding vector
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if item_id not in item_mapping:
            return None
        
        item_idx = item_mapping[item_id]
        return self.model.item_factors[item_idx]
