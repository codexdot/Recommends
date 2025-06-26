import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import precision_score, recall_score
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommendationEvaluator:
    """
    Evaluation framework for recommendation systems.
    """
    
    def __init__(self):
        pass
    
    def evaluate_model(self, recommender, train_matrix, test_matrix, 
                      user_mapping, item_mapping, k_values=[5, 10, 20]):
        """
        Comprehensive evaluation of the recommendation model.
        
        Args:
            recommender: Trained recommendation model
            train_matrix (csr_matrix): Training user-item matrix
            test_matrix (csr_matrix): Test user-item matrix
            user_mapping (dict): User ID to index mapping
            item_mapping (dict): Item ID to index mapping
            k_values (list): List of k values for evaluation
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        logger.info("Starting model evaluation...")
        
        results = {}
        
        # Get test users (users who have interactions in test set)
        test_users = set(test_matrix.nonzero()[0])
        logger.info(f"Evaluating on {len(test_users)} test users")
        
        # Calculate metrics for each k value
        for k in k_values:
            logger.info(f"Calculating metrics for k={k}")
            
            precision_scores = []
            recall_scores = []
            ndcg_scores = []
            
            for user_idx in test_users:
                # Get test items for this user
                test_items = set(test_matrix[user_idx].nonzero()[1])
                
                if len(test_items) == 0:
                    continue
                
                # Generate recommendations
                try:
                    # Get user ID from mapping
                    reverse_user_mapping = {idx: user for user, idx in user_mapping.items()}
                    user_id = reverse_user_mapping[user_idx]
                    
                    recommendations = recommender.recommend_for_user(
                        user_id=user_id,
                        user_item_matrix=train_matrix,
                        user_mapping=user_mapping,
                        item_mapping=item_mapping,
                        n_recommendations=k,
                        include_explanations=False
                    )
                    
                    # Extract recommended item indices
                    recommended_items = set()
                    recommendation_scores = {}
                    
                    for item_id, score, _ in recommendations:
                        if item_id in item_mapping:
                            item_idx = item_mapping[item_id]
                            recommended_items.add(item_idx)
                            recommendation_scores[item_idx] = score
                    
                    # Calculate metrics
                    if len(recommended_items) > 0:
                        # Precision@k
                        precision = len(test_items.intersection(recommended_items)) / len(recommended_items)
                        precision_scores.append(precision)
                        
                        # Recall@k
                        recall = len(test_items.intersection(recommended_items)) / len(test_items)
                        recall_scores.append(recall)
                        
                        # NDCG@k
                        ndcg = self._calculate_ndcg_at_k(
                            test_items, recommended_items, recommendation_scores, k
                        )
                        ndcg_scores.append(ndcg)
                
                except Exception as e:
                    logger.warning(f"Error evaluating user {user_idx}: {e}")
                    continue
            
            # Average metrics
            results[f'precision_at_{k}'] = np.mean(precision_scores) if precision_scores else 0.0
            results[f'recall_at_{k}'] = np.mean(recall_scores) if recall_scores else 0.0
            results[f'ndcg_at_{k}'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
            
            logger.info(f"k={k}: Precision={results[f'precision_at_{k}']:.4f}, "
                       f"Recall={results[f'recall_at_{k}']:.4f}, "
                       f"NDCG={results[f'ndcg_at_{k}']:.4f}")
        
        # Calculate coverage and diversity
        results['coverage'] = self._calculate_coverage(
            recommender, train_matrix, user_mapping, item_mapping
        )
        
        results['diversity'] = self._calculate_diversity(
            recommender, train_matrix, user_mapping, item_mapping
        )
        
        logger.info("Model evaluation completed")
        return results
    
    def _calculate_ndcg_at_k(self, relevant_items, recommended_items, scores, k):
        """
        Calculate Normalized Discounted Cumulative Gain at k.
        
        Args:
            relevant_items (set): Set of relevant item indices
            recommended_items (set): Set of recommended item indices
            scores (dict): Dictionary of item_idx -> score
            k (int): Cutoff value
            
        Returns:
            float: NDCG@k score
        """
        # Create ordered list of recommendations by score
        ordered_recommendations = sorted(
            [(item_idx, scores.get(item_idx, 0)) for item_idx in recommended_items],
            key=lambda x: x[1], reverse=True
        )[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, (item_idx, score) in enumerate(ordered_recommendations):
            if item_idx in relevant_items:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(len(relevant_items), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        # Return NDCG
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_coverage(self, recommender, train_matrix, user_mapping, item_mapping, n_users=100):
        """
        Calculate item coverage - percentage of items that appear in recommendations.
        
        Args:
            recommender: Trained recommendation model
            train_matrix (csr_matrix): Training user-item matrix
            user_mapping (dict): User ID to index mapping
            item_mapping (dict): Item ID to index mapping
            n_users (int): Number of users to sample for coverage calculation
            
        Returns:
            float: Coverage percentage
        """
        total_items = len(item_mapping)
        recommended_items = set()
        
        # Sample users for coverage calculation
        reverse_user_mapping = {idx: user for user, idx in user_mapping.items()}
        sample_users = np.random.choice(
            list(user_mapping.keys()), 
            size=min(n_users, len(user_mapping)), 
            replace=False
        )
        
        for user_id in sample_users:
            try:
                recommendations = recommender.recommend_for_user(
                    user_id=user_id,
                    user_item_matrix=train_matrix,
                    user_mapping=user_mapping,
                    item_mapping=item_mapping,
                    n_recommendations=20,
                    include_explanations=False
                )
                
                for item_id, _, _ in recommendations:
                    if item_id in item_mapping:
                        recommended_items.add(item_id)
                        
            except Exception as e:
                logger.warning(f"Error calculating coverage for user {user_id}: {e}")
                continue
        
        coverage = len(recommended_items) / total_items if total_items > 0 else 0.0
        return coverage
    
    def _calculate_diversity(self, recommender, train_matrix, user_mapping, item_mapping, n_users=50):
        """
        Calculate intra-list diversity - average diversity within recommendation lists.
        
        Args:
            recommender: Trained recommendation model
            train_matrix (csr_matrix): Training user-item matrix
            user_mapping (dict): User ID to index mapping
            item_mapping (dict): Item ID to index mapping
            n_users (int): Number of users to sample for diversity calculation
            
        Returns:
            float: Average diversity score
        """
        diversity_scores = []
        
        # Sample users for diversity calculation
        sample_users = np.random.choice(
            list(user_mapping.keys()), 
            size=min(n_users, len(user_mapping)), 
            replace=False
        )
        
        for user_id in sample_users:
            try:
                recommendations = recommender.recommend_for_user(
                    user_id=user_id,
                    user_item_matrix=train_matrix,
                    user_mapping=user_mapping,
                    item_mapping=item_mapping,
                    n_recommendations=10,
                    include_explanations=False
                )
                
                if len(recommendations) < 2:
                    continue
                
                # Calculate pairwise diversity
                total_pairs = 0
                diverse_pairs = 0
                
                for i in range(len(recommendations)):
                    for j in range(i + 1, len(recommendations)):
                        item1_id, _, _ = recommendations[i]
                        item2_id, _, _ = recommendations[j]
                        
                        # Simple diversity measure: items are diverse if they're different
                        if item1_id != item2_id:
                            diverse_pairs += 1
                        total_pairs += 1
                
                if total_pairs > 0:
                    diversity = diverse_pairs / total_pairs
                    diversity_scores.append(diversity)
                    
            except Exception as e:
                logger.warning(f"Error calculating diversity for user {user_id}: {e}")
                continue
        
        return np.mean(diversity_scores) if diversity_scores else 0.0
    
    def evaluate_cold_start(self, recommender, train_matrix, test_matrix, 
                           user_mapping, item_mapping, cold_start_threshold=5):
        """
        Evaluate performance on cold start users.
        
        Args:
            recommender: Trained recommendation model
            train_matrix (csr_matrix): Training user-item matrix
            test_matrix (csr_matrix): Test user-item matrix
            user_mapping (dict): User ID to index mapping
            item_mapping (dict): Item ID to index mapping
            cold_start_threshold (int): Maximum interactions to be considered cold start
            
        Returns:
            dict: Cold start evaluation results
        """
        logger.info("Evaluating cold start performance...")
        
        # Identify cold start users in training data
        user_interaction_counts = np.array(train_matrix.sum(axis=1)).flatten()
        cold_start_user_indices = np.where(user_interaction_counts <= cold_start_threshold)[0]
        
        # Filter to users who also have test interactions
        test_users = set(test_matrix.nonzero()[0])
        cold_start_test_users = [idx for idx in cold_start_user_indices if idx in test_users]
        
        logger.info(f"Found {len(cold_start_test_users)} cold start users in test set")
        
        if len(cold_start_test_users) == 0:
            return {'cold_start_precision': 0.0, 'cold_start_recall': 0.0, 'cold_start_coverage': 0.0}
        
        precision_scores = []
        recall_scores = []
        recommended_items = set()
        
        reverse_user_mapping = {idx: user for user, idx in user_mapping.items()}
        
        for user_idx in cold_start_test_users:
            try:
                user_id = reverse_user_mapping[user_idx]
                test_items = set(test_matrix[user_idx].nonzero()[1])
                
                if len(test_items) == 0:
                    continue
                
                recommendations = recommender.recommend_for_user(
                    user_id=user_id,
                    user_item_matrix=train_matrix,
                    user_mapping=user_mapping,
                    item_mapping=item_mapping,
                    n_recommendations=10,
                    include_explanations=False
                )
                
                rec_items = set()
                for item_id, _, _ in recommendations:
                    if item_id in item_mapping:
                        item_idx = item_mapping[item_id]
                        rec_items.add(item_idx)
                        recommended_items.add(item_id)
                
                if len(rec_items) > 0:
                    precision = len(test_items.intersection(rec_items)) / len(rec_items)
                    recall = len(test_items.intersection(rec_items)) / len(test_items)
                    
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    
            except Exception as e:
                logger.warning(f"Error evaluating cold start user {user_idx}: {e}")
                continue
        
        results = {
            'cold_start_precision': np.mean(precision_scores) if precision_scores else 0.0,
            'cold_start_recall': np.mean(recall_scores) if recall_scores else 0.0,
            'cold_start_coverage': len(recommended_items) / len(item_mapping) if item_mapping else 0.0
        }
        
        logger.info(f"Cold start evaluation: Precision={results['cold_start_precision']:.4f}, "
                   f"Recall={results['cold_start_recall']:.4f}, "
                   f"Coverage={results['cold_start_coverage']:.4f}")
        
        return results
