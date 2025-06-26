import numpy as np
from scipy.sparse import csr_matrix
import implicit
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization
import logging
import pickle
import json
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImplicitRecommender:
    """
    Wrapper class for implicit feedback recommendation algorithms.
    """
    
    def __init__(self, algorithm='als', factors=50, regularization=0.1, 
                 iterations=20, alpha=15, random_state=42):
        """
        Initialize the recommender.
        
        Args:
            algorithm (str): Algorithm to use ('als', 'bpr', 'lmf')
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
        
        # Initialize the model based on algorithm
        if algorithm == 'als':
            self.model = AlternatingLeastSquares(
                factors=factors,
                regularization=regularization,
                iterations=iterations,
                alpha=alpha,
                random_state=random_state
            )
        elif algorithm == 'bpr':
            self.model = BayesianPersonalizedRanking(
                factors=factors,
                regularization=regularization,
                iterations=iterations,
                random_state=random_state
            )
        elif algorithm == 'lmf':
            self.model = LogisticMatrixFactorization(
                factors=factors,
                regularization=regularization,
                iterations=iterations,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        self.is_fitted = False
        logger.info(f"Initialized {algorithm} recommender with {factors} factors")
    
    def fit(self, user_item_matrix):
        """
        Train the recommendation model.
        
        Args:
            user_item_matrix (csr_matrix): User-item interaction matrix
        """
        logger.info("Training recommendation model...")
        
        # Ensure matrix is in the right format (items x users for implicit library)
        item_user_matrix = user_item_matrix.T.tocsr()
        
        # Fit the model
        self.model.fit(item_user_matrix, show_progress=True)
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
        
        # Validate user index bounds
        if user_idx >= user_item_matrix.shape[0]:
            logger.error(f"User index {user_idx} out of bounds for matrix shape {user_item_matrix.shape}")
            return self._get_cold_start_recommendations(
                user_item_matrix, item_mapping, n_recommendations, include_explanations
            )
        
        # Get recommendations from the model
        item_user_matrix = user_item_matrix.T.tocsr()
        
        try:
            # Convert numpy types to Python native types for compatibility
            user_idx_int = int(user_idx)
            
            # Validate that user_idx is within bounds
            if user_idx_int >= user_item_matrix.shape[0]:
                logger.error(f"User index {user_idx_int} out of bounds for matrix shape {user_item_matrix.shape}")
                return self._get_cold_start_recommendations(
                    user_item_matrix, item_mapping, n_recommendations, include_explanations
                )
            
            recommended_items, scores = self.model.recommend(
                user_idx_int, 
                user_item_matrix[user_idx_int],
                N=n_recommendations,
                filter_already_liked_items=True
            )
            
            # Validate recommended item indices against actual matrix bounds
            max_item_idx = user_item_matrix.shape[1] - 1
            reverse_item_mapping = {idx: item for item, idx in item_mapping.items()}
            
            valid_recommendations = []
            valid_scores = []
            
            for item_idx, score in zip(recommended_items, scores):
                if 0 <= item_idx <= max_item_idx and item_idx in reverse_item_mapping:
                    valid_recommendations.append(item_idx)
                    valid_scores.append(score)
                else:
                    logger.warning(f"Item index {item_idx} out of bounds (max: {max_item_idx}) or not in mapping")
            
            recommended_items = np.array(valid_recommendations)
            scores = np.array(valid_scores)
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return self._get_cold_start_recommendations(
                user_item_matrix, item_mapping, n_recommendations, include_explanations
            )
        
        # Map back to original item IDs
        reverse_item_mapping = {idx: item for item, idx in item_mapping.items()}
        
        recommendations = []
        for item_idx, score in zip(recommended_items, scores):
            # Convert numpy types to Python native types for compatibility
            item_idx_int = int(item_idx)
            
            # Check if item index is valid
            if item_idx_int not in reverse_item_mapping:
                logger.warning(f"Item index {item_idx_int} not found in mapping, skipping")
                continue
                
            item_id = reverse_item_mapping[item_idx_int]
            
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
        Enhanced cold start recommendations using multiple strategies.
        
        Args:
            user_item_matrix (csr_matrix): User-item interaction matrix
            item_mapping (dict): Item ID to index mapping
            n_recommendations (int): Number of recommendations
            include_explanations (bool): Whether to include explanations
            
        Returns:
            list: List of (item_id, score, explanation) tuples
        """
        # Strategy 1: Item popularity
        item_popularity = np.array(user_item_matrix.sum(axis=0)).flatten()
        
        # Strategy 2: Item diversity (items with balanced user engagement)
        item_user_counts = np.array((user_item_matrix > 0).sum(axis=0)).flatten()
        item_avg_rating = np.divide(item_popularity, item_user_counts, 
                                   out=np.zeros_like(item_popularity), 
                                   where=item_user_counts!=0)
        
        # Strategy 3: Recent popularity (if we had timestamps, but for now use recency proxy)
        # Use items that appear in the latter part of the matrix as a proxy
        n_users = user_item_matrix.shape[0]
        recent_users = user_item_matrix[int(n_users*0.7):, :]  # Last 30% of users
        recent_popularity = np.array(recent_users.sum(axis=0)).flatten()
        
        # Combine strategies with weights
        popularity_weight = 0.5
        diversity_weight = 0.3
        recency_weight = 0.2
        
        # Normalize each score to [0, 1]
        norm_popularity = item_popularity / (np.max(item_popularity) + 1e-8)
        norm_diversity = item_avg_rating / (np.max(item_avg_rating) + 1e-8)
        norm_recency = recent_popularity / (np.max(recent_popularity) + 1e-8)
        
        # Combined score
        combined_score = (popularity_weight * norm_popularity + 
                         diversity_weight * norm_diversity + 
                         recency_weight * norm_recency)
        
        # Get top items based on combined score
        top_item_indices = np.argsort(combined_score)[::-1][:n_recommendations]
        
        reverse_item_mapping = {idx: item for item, idx in item_mapping.items()}
        
        recommendations = []
        for item_idx in top_item_indices:
            # Ensure item_idx is valid
            if item_idx >= len(reverse_item_mapping):
                continue
                
            if item_idx not in reverse_item_mapping:
                continue
                
            item_id = reverse_item_mapping[item_idx]
            score = combined_score[item_idx]
            
            explanation = None
            if include_explanations:
                pop_count = int(item_popularity[item_idx])
                user_count = int(item_user_counts[item_idx])
                avg_rating = item_avg_rating[item_idx]
                
                explanation = (f"Cold start recommendation: Popular item "
                             f"({pop_count} interactions from {user_count} users, "
                             f"avg rating: {avg_rating:.2f})")
            
            recommendations.append((item_id, float(score), explanation))
        
        return recommendations
    
    def _generate_explanation(self, user_id, item_id, user_item_matrix, 
                            user_mapping, item_mapping):
        """
        Generate detailed explanation for a recommendation.
        
        Args:
            user_id: Original user ID
            item_id: Original item ID
            user_item_matrix (csr_matrix): User-item interaction matrix
            user_mapping (dict): User ID to index mapping
            item_mapping (dict): Item ID to index mapping
            
        Returns:
            str: Detailed explanation text
        """
        try:
            user_idx = user_mapping[user_id]
            item_idx = item_mapping[item_id]
            
            # Get user's interaction history
            user_interactions = user_item_matrix[user_idx].nonzero()[1]
            user_total_interactions = user_item_matrix[user_idx].sum()
            
            # Get item popularity metrics
            item_popularity = user_item_matrix[:, item_idx].sum()
            item_user_count = user_item_matrix[:, item_idx].nnz
            
            # Calculate item category (simulated based on item ID ranges)
            n_items = user_item_matrix.shape[1]
            category_size = n_items // 5
            item_category = min(item_idx // category_size, 4)
            category_names = ["Electronics", "Books", "Movies", "Fashion", "Home"]
            
            # Find similar items the user has interacted with
            similar_items = []
            for user_item_idx in user_interactions[-5:]:  # Last 5 items
                if abs(user_item_idx - item_idx) <= 20:  # Items within similar range
                    similar_items.append(user_item_idx)
            
            # Generate explanation based on different factors
            explanations = []
            
            # Popularity explanation
            if item_popularity > np.percentile(user_item_matrix.sum(axis=0), 80):
                explanations.append(f"highly popular item ({item_user_count} users)")
            elif item_popularity > np.percentile(user_item_matrix.sum(axis=0), 60):
                explanations.append(f"moderately popular item ({item_user_count} users)")
            
            # Category explanation
            user_category_interactions = sum(1 for idx in user_interactions 
                                           if idx // category_size == item_category)
            if user_category_interactions > 0:
                explanations.append(f"matches your {category_names[item_category]} preferences ({user_category_interactions} similar items)")
            
            # Similar items explanation
            if similar_items:
                explanations.append(f"similar to {len(similar_items)} items you've liked")
            
            # User behavior explanation
            if user_total_interactions > 20:
                explanations.append("recommended for active users like you")
            elif user_total_interactions > 10:
                explanations.append("good match for your activity level")
            
            # Combine explanations
            if explanations:
                return f"Recommended because it's a {', '.join(explanations[:2])}"
            else:
                return f"Recommended based on collaborative filtering analysis"
                
        except Exception as e:
            logger.warning(f"Error generating explanation: {e}")
            return "Recommended by our AI system based on user preferences"
            
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
    
    def save_model(self, filepath=None):
        """
        Save the trained model and its configuration to disk.
        
        Args:
            filepath (str): Path to save the model (optional)
            
        Returns:
            str: Path to the saved model file
        """
        if not self.model_trained:
            raise ValueError("No trained model to save. Please train the model first.")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"recommendation_model_{timestamp}.pkl"
        
        # Create model data structure
        model_data = {
            'algorithm': self.algorithm,
            'factors': self.factors,
            'regularization': self.regularization,
            'iterations': self.iterations,
            'alpha': self.alpha,
            'random_state': self.random_state,
            'model_trained': self.model_trained,
            'model': self.model,
            'save_timestamp': datetime.now().isoformat(),
            'model_type': type(self.model).__name__
        }
        
        # Save to pickle file
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath):
        """
        Load a previously trained model from disk.
        
        Args:
            filepath (str): Path to the saved model file
            
        Returns:
            dict: Model metadata
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load from pickle file
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore model parameters
        self.algorithm = model_data['algorithm']
        self.factors = model_data['factors']
        self.regularization = model_data['regularization']
        self.iterations = model_data['iterations']
        self.alpha = model_data['alpha']
        self.random_state = model_data['random_state']
        self.model_trained = model_data['model_trained']
        self.model = model_data['model']
        
        logger.info(f"Model loaded from {filepath}")
        
        # Return metadata
        return {
            'algorithm': self.algorithm,
            'factors': self.factors,
            'save_timestamp': model_data.get('save_timestamp'),
            'model_type': model_data.get('model_type'),
            'filepath': filepath
        }
    
    def export_model_info(self):
        """
        Export model configuration and statistics as JSON.
        
        Returns:
            dict: Model information
        """
        if not self.model_trained:
            return {'error': 'No trained model available'}
        
        model_info = {
            'algorithm': self.algorithm,
            'hyperparameters': {
                'factors': self.factors,
                'regularization': self.regularization,
                'iterations': self.iterations,
                'alpha': self.alpha,
                'random_state': self.random_state
            },
            'model_size_mb': self.get_model_size(),
            'model_type': type(self.model).__name__,
            'export_timestamp': datetime.now().isoformat()
        }
        
        return model_info
