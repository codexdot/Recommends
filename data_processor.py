import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Handles data preprocessing for implicit feedback recommendation systems.
    """
    
    def __init__(self):
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
    
    def create_user_item_matrix(self, interactions_df, min_interactions_per_user=1, min_interactions_per_item=1):
        """
        Create user-item interaction matrix from interaction data.
        
        Args:
            interactions_df (pd.DataFrame): DataFrame with columns [user_id, item_id, rating]
            min_interactions_per_user (int): Minimum interactions required per user
            min_interactions_per_item (int): Minimum interactions required per item
            
        Returns:
            tuple: (user_item_matrix, user_mapping, item_mapping)
        """
        logger.info("Creating user-item matrix...")
        
        # Filter users and items with minimum interactions
        user_counts = interactions_df['user_id'].value_counts()
        item_counts = interactions_df['item_id'].value_counts()
        
        valid_users = user_counts[user_counts >= min_interactions_per_user].index
        valid_items = item_counts[item_counts >= min_interactions_per_item].index
        
        # Filter dataframe
        filtered_df = interactions_df[
            (interactions_df['user_id'].isin(valid_users)) &
            (interactions_df['item_id'].isin(valid_items))
        ].copy()
        
        logger.info(f"Filtered data: {len(filtered_df)} interactions, "
                   f"{filtered_df['user_id'].nunique()} users, "
                   f"{filtered_df['item_id'].nunique()} items")
        
        # Create mappings
        unique_users = sorted(filtered_df['user_id'].unique())
        unique_items = sorted(filtered_df['item_id'].unique())
        
        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}
        self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}
        
        # Map IDs to indices
        filtered_df['user_idx'] = filtered_df['user_id'].map(self.user_mapping)
        filtered_df['item_idx'] = filtered_df['item_id'].map(self.item_mapping)
        
        # Create sparse matrix
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        # Handle duplicate interactions by summing ratings
        interaction_data = filtered_df.groupby(['user_idx', 'item_idx'])['rating'].sum().reset_index()
        
        user_item_matrix = csr_matrix(
            (interaction_data['rating'], 
             (interaction_data['user_idx'], interaction_data['item_idx'])),
            shape=(n_users, n_items)
        )
        
        logger.info(f"Created user-item matrix: {user_item_matrix.shape}")
        logger.info(f"Matrix sparsity: {1 - user_item_matrix.nnz / (n_users * n_items):.4f}")
        
        return user_item_matrix, self.user_mapping, self.item_mapping
    
    def train_test_split(self, user_item_matrix, test_size=0.2, random_state=42):
        """
        Split user-item matrix into train and test sets.
        
        Args:
            user_item_matrix (csr_matrix): User-item interaction matrix
            test_size (float): Proportion of interactions for testing
            random_state (int): Random seed
            
        Returns:
            tuple: (train_matrix, test_matrix)
        """
        logger.info(f"Splitting data with test_size={test_size}")
        
        # Get non-zero entries
        user_indices, item_indices = user_item_matrix.nonzero()
        ratings = user_item_matrix.data
        
        # Split indices - use simple random split to avoid stratification issues
        # with users who have very few interactions
        train_idx, test_idx = train_test_split(
            range(len(user_indices)),
            test_size=test_size,
            random_state=random_state
        )
        
        # Create train matrix
        train_users = user_indices[train_idx]
        train_items = item_indices[train_idx]
        train_ratings = ratings[train_idx]
        
        train_matrix = csr_matrix(
            (train_ratings, (train_users, train_items)),
            shape=user_item_matrix.shape
        )
        
        # Create test matrix
        test_users = user_indices[test_idx]
        test_items = item_indices[test_idx]
        test_ratings = ratings[test_idx]
        
        test_matrix = csr_matrix(
            (test_ratings, (test_users, test_items)),
            shape=user_item_matrix.shape
        )
        
        logger.info(f"Train matrix: {train_matrix.nnz} interactions")
        logger.info(f"Test matrix: {test_matrix.nnz} interactions")
        
        return train_matrix, test_matrix
    
    def apply_implicit_weighting(self, user_item_matrix, alpha=15):
        """
        Apply implicit feedback weighting to the user-item matrix.
        
        Args:
            user_item_matrix (csr_matrix): User-item interaction matrix
            alpha (float): Confidence parameter
            
        Returns:
            csr_matrix: Weighted matrix
        """
        logger.info(f"Applying implicit weighting with alpha={alpha}")
        
        # Create confidence matrix: C = 1 + alpha * R
        confidence_matrix = user_item_matrix.copy()
        confidence_matrix.data = 1 + alpha * confidence_matrix.data
        
        return confidence_matrix
    
    def get_user_items(self, user_id, user_item_matrix, user_mapping, item_mapping):
        """
        Get items interacted with by a specific user.
        
        Args:
            user_id: Original user ID
            user_item_matrix (csr_matrix): User-item interaction matrix
            user_mapping (dict): User ID to index mapping
            item_mapping (dict): Item ID to index mapping
            
        Returns:
            list: List of (item_id, rating) tuples
        """
        if user_id not in user_mapping:
            return []
        
        user_idx = user_mapping[user_id]
        user_items = user_item_matrix[user_idx].nonzero()[1]
        user_ratings = user_item_matrix[user_idx].data
        
        reverse_item_mapping = {idx: item for item, idx in item_mapping.items()}
        
        return [(reverse_item_mapping[item_idx], rating) 
                for item_idx, rating in zip(user_items, user_ratings)]
    
    def get_popular_items(self, user_item_matrix, item_mapping, n_items=10):
        """
        Get most popular items based on total interactions.
        
        Args:
            user_item_matrix (csr_matrix): User-item interaction matrix
            item_mapping (dict): Item ID to index mapping
            n_items (int): Number of popular items to return
            
        Returns:
            list: List of (item_id, popularity_score) tuples
        """
        # Calculate item popularity (sum of all interactions)
        item_popularity = np.array(user_item_matrix.sum(axis=0)).flatten()
        
        # Get top items
        top_item_indices = np.argsort(item_popularity)[::-1][:n_items]
        
        reverse_item_mapping = {idx: item for item, idx in item_mapping.items()}
        
        popular_items = [
            (reverse_item_mapping[item_idx], item_popularity[item_idx])
            for item_idx in top_item_indices
        ]
        
        return popular_items
