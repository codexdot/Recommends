import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_data(n_users=2000, n_items=4500, n_interactions=50000, 
                        sparsity=0.95, random_state=42):
    """
    Generate synthetic implicit feedback data for demonstration.
    
    Args:
        n_users (int): Number of users
        n_items (int): Number of items
        n_interactions (int): Number of interactions
        sparsity (float): Sparsity level (0-1)
        random_state (int): Random seed
        
    Returns:
        pd.DataFrame: Synthetic interaction data
    """
    logger.info(f"Generating sample data: {n_users} users, {n_items} items, {n_interactions} interactions")
    
    np.random.seed(random_state)
    
    # Generate user and item IDs
    user_ids = np.arange(1, n_users + 1)
    item_ids = np.arange(1, n_items + 1)
    
    # Create realistic interaction patterns
    interactions = []
    
    # Popular items (20% of items get 60% of interactions)
    popular_items = np.random.choice(item_ids, size=int(0.2 * n_items), replace=False)
    regular_items = np.setdiff1d(item_ids, popular_items)
    
    # Active users (50% of users generate 80% of interactions) - reduces cold users
    active_users = np.random.choice(user_ids, size=int(0.5 * n_users), replace=False)
    regular_users = np.setdiff1d(user_ids, active_users)
    
    interactions_generated = 0
    
    # Generate interactions for active users with popular items
    while interactions_generated < int(0.8 * n_interactions):
        user_id = np.random.choice(active_users)
        item_id = np.random.choice(popular_items)
        
        # Implicit feedback scores (higher for popular items)
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.3, 0.3, 0.1])
        
        interactions.append({
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating,
            'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 365))
        })
        
        interactions_generated += 1
    
    # Generate remaining interactions (focus on regular users to reduce cold start)
    while interactions_generated < n_interactions:
        # 70% chance to pick from regular users, 30% from all users
        if np.random.random() < 0.7:
            user_id = np.random.choice(regular_users)
        else:
            user_id = np.random.choice(user_ids)
        
        item_id = np.random.choice(item_ids)
        
        # Slightly higher ratings for regular interactions to reduce cold users
        rating = np.random.choice([1, 2, 3, 4], p=[0.3, 0.3, 0.3, 0.1])
        
        interactions.append({
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating,
            'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 365))
        })
        
        interactions_generated += 1
    
    # Create DataFrame
    df = pd.DataFrame(interactions)
    
    # Remove duplicates (keep highest rating for user-item pairs)
    df = df.groupby(['user_id', 'item_id']).agg({
        'rating': 'max',
        'timestamp': 'first'
    }).reset_index()
    
    # Add some noise and variation
    df['rating'] = df['rating'] + np.random.normal(0, 0.1, len(df))
    df['rating'] = np.clip(df['rating'], 0.1, 5.0)
    
    logger.info(f"Generated {len(df)} unique interactions")
    
    return df[['user_id', 'item_id', 'rating']]

def format_recommendations(recommendations, include_scores=True):
    """
    Format recommendations for display.
    
    Args:
        recommendations (list): List of (item_id, score, explanation) tuples
        include_scores (bool): Whether to include scores in formatting
        
    Returns:
        str: Formatted recommendations string
    """
    if not recommendations:
        return "No recommendations available."
    
    formatted = []
    for i, (item_id, score, explanation) in enumerate(recommendations, 1):
        if include_scores:
            line = f"{i}. Item {item_id} (Score: {score:.3f})"
        else:
            line = f"{i}. Item {item_id}"
        
        if explanation:
            line += f" - {explanation}"
        
        formatted.append(line)
    
    return "\n".join(formatted)

def calculate_user_similarity(user1_items, user2_items):
    """
    Calculate Jaccard similarity between two users' item sets.
    
    Args:
        user1_items (set): Set of items for user 1
        user2_items (set): Set of items for user 2
        
    Returns:
        float: Jaccard similarity score
    """
    if not user1_items or not user2_items:
        return 0.0
    
    intersection = len(user1_items.intersection(user2_items))
    union = len(user1_items.union(user2_items))
    
    return intersection / union if union > 0 else 0.0

def calculate_item_similarity(user_item_matrix, item1_idx, item2_idx):
    """
    Calculate cosine similarity between two items based on user interactions.
    
    Args:
        user_item_matrix (csr_matrix): User-item interaction matrix
        item1_idx (int): Index of first item
        item2_idx (int): Index of second item
        
    Returns:
        float: Cosine similarity score
    """
    item1_vector = user_item_matrix[:, item1_idx].toarray().flatten()
    item2_vector = user_item_matrix[:, item2_idx].toarray().flatten()
    
    # Calculate cosine similarity
    dot_product = np.dot(item1_vector, item2_vector)
    norm1 = np.linalg.norm(item1_vector)
    norm2 = np.linalg.norm(item2_vector)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def get_data_statistics(interactions_df):
    """
    Calculate comprehensive statistics for interaction data.
    
    Args:
        interactions_df (pd.DataFrame): Interaction data
        
    Returns:
        dict: Dictionary of statistics
    """
    stats = {}
    
    # Basic counts
    stats['total_interactions'] = len(interactions_df)
    stats['unique_users'] = interactions_df['user_id'].nunique()
    stats['unique_items'] = interactions_df['item_id'].nunique()
    
    # Sparsity
    possible_interactions = stats['unique_users'] * stats['unique_items']
    stats['sparsity'] = 1 - (stats['total_interactions'] / possible_interactions)
    
    # User statistics
    user_interactions = interactions_df['user_id'].value_counts()
    stats['avg_interactions_per_user'] = user_interactions.mean()
    stats['median_interactions_per_user'] = user_interactions.median()
    stats['max_interactions_per_user'] = user_interactions.max()
    stats['min_interactions_per_user'] = user_interactions.min()
    
    # Item statistics
    item_interactions = interactions_df['item_id'].value_counts()
    stats['avg_interactions_per_item'] = item_interactions.mean()
    stats['median_interactions_per_item'] = item_interactions.median()
    stats['max_interactions_per_item'] = item_interactions.max()
    stats['min_interactions_per_item'] = item_interactions.min()
    
    # Rating statistics
    stats['avg_rating'] = interactions_df['rating'].mean()
    stats['median_rating'] = interactions_df['rating'].median()
    stats['std_rating'] = interactions_df['rating'].std()
    
    return stats

def validate_interaction_data(interactions_df):
    """
    Validate interaction data format and quality.
    
    Args:
        interactions_df (pd.DataFrame): Interaction data to validate
        
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    
    # Check required columns
    required_columns = ['user_id', 'item_id', 'rating']
    missing_columns = [col for col in required_columns if col not in interactions_df.columns]
    
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
        return False, errors
    
    # Check for empty dataframe
    if len(interactions_df) == 0:
        errors.append("Dataset is empty")
        return False, errors
    
    # Check for null values
    null_counts = interactions_df[required_columns].isnull().sum()
    if null_counts.any():
        errors.append(f"Null values found: {null_counts.to_dict()}")
    
    # Check data types
    if not pd.api.types.is_numeric_dtype(interactions_df['rating']):
        errors.append("Rating column must be numeric")
    
    # Check rating range
    if interactions_df['rating'].min() < 0:
        errors.append("Ratings must be non-negative for implicit feedback")
    
    # Check for sufficient data
    n_users = interactions_df['user_id'].nunique()
    n_items = interactions_df['item_id'].nunique()
    
    if n_users < 2:
        errors.append("Dataset must contain at least 2 unique users")
    
    if n_items < 2:
        errors.append("Dataset must contain at least 2 unique items")
    
    # Check sparsity
    sparsity = 1 - (len(interactions_df) / (n_users * n_items))
    if sparsity > 0.999:
        errors.append(f"Dataset is too sparse ({sparsity:.1%}). Consider reducing users/items or adding more interactions.")
    
    is_valid = len(errors) == 0
    return is_valid, errors

def export_recommendations(user_recommendations, filename=None):
    """
    Export recommendations to CSV format.
    
    Args:
        user_recommendations (dict): Dictionary of user_id -> recommendations
        filename (str): Output filename (optional)
        
    Returns:
        pd.DataFrame: DataFrame with recommendations
    """
    export_data = []
    
    for user_id, recommendations in user_recommendations.items():
        for rank, (item_id, score, explanation) in enumerate(recommendations, 1):
            export_data.append({
                'user_id': user_id,
                'rank': rank,
                'item_id': item_id,
                'score': score,
                'explanation': explanation or ''
            })
    
    df = pd.DataFrame(export_data)
    
    if filename:
        df.to_csv(filename, index=False)
        logger.info(f"Recommendations exported to {filename}")
    
    return df
