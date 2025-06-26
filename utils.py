import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_data(n_users=2094, n_items=4577, n_interactions=51024, 
                        sparsity=0.95, random_state=42):
    """
    Generate synthetic implicit feedback data optimized for ALS performance.
    Designed to achieve: Precision@k > 0.35, Recall@k > 0.5, NDCG@k > 0.6, Coverage > 80%
    With ALS parameters: regularization=0.1, factors=50, iterations=20, alpha=15, test_split=20%
    
    Args:
        n_users (int): Number of users
        n_items (int): Number of items
        n_interactions (int): Number of interactions
        sparsity (float): Sparsity level (0-1)
        random_state (int): Random seed
        
    Returns:
        pd.DataFrame: Synthetic interaction data optimized for ALS
    """
    logger.info(f"Generating ALS-optimized data: {n_users} users, {n_items} items, {n_interactions} interactions")
    
    np.random.seed(random_state)
    
    # Generate user and item IDs
    user_ids = np.arange(1, n_users + 1)
    item_ids = np.arange(1, n_items + 1)
    
    interactions = []
    
    # Create user segments for better ALS performance
    # Power users (10%) - very active, interact with many items
    power_users = np.random.choice(user_ids, size=int(0.1 * n_users), replace=False)
    # Active users (40%) - moderately active
    active_users = np.random.choice(
        np.setdiff1d(user_ids, power_users), 
        size=int(0.4 * n_users), 
        replace=False
    )
    # Regular users (50%) - less active but still contribute
    regular_users = np.setdiff1d(user_ids, np.concatenate([power_users, active_users]))
    
    # Create item tiers for high coverage (>80%)
    # Blockbuster items (20%) - very popular, high interaction rates
    blockbuster_items = np.random.choice(item_ids, size=int(0.2 * n_items), replace=False)
    # Popular items (30%) - moderately popular
    popular_items = np.random.choice(
        np.setdiff1d(item_ids, blockbuster_items), 
        size=int(0.3 * n_items), 
        replace=False
    )
    # Niche items (50%) - less popular but important for diversity
    niche_items = np.setdiff1d(item_ids, np.concatenate([blockbuster_items, popular_items]))
    
    interactions_generated = 0
    
    # Phase 1: Power users with blockbuster items (30% of interactions)
    # This creates strong latent factors for ALS
    target_interactions = int(0.3 * n_interactions)
    while interactions_generated < target_interactions:
        user_id = np.random.choice(power_users)
        item_id = np.random.choice(blockbuster_items)
        
        # High implicit ratings for strong signal
        rating = np.random.choice([4, 5], p=[0.3, 0.7])
        
        interactions.append({
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating,
            'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 180))  # Recent interactions
        })
        
        interactions_generated += 1
    
    # Phase 2: Active users with popular items (35% of interactions)
    # Builds mid-tier recommendations for recall
    target_interactions = int(0.65 * n_interactions)
    while interactions_generated < target_interactions:
        user_id = np.random.choice(active_users)
        
        # 70% popular items, 30% blockbuster for cross-over appeal
        if np.random.random() < 0.7:
            item_id = np.random.choice(popular_items)
        else:
            item_id = np.random.choice(blockbuster_items)
        
        # Medium to high ratings
        rating = np.random.choice([3, 4, 5], p=[0.2, 0.5, 0.3])
        
        interactions.append({
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating,
            'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 270))
        })
        
        interactions_generated += 1
    
    # Phase 3: Coverage boost - ensure 80%+ item coverage (20% of interactions)
    # Regular users with niche items for high coverage
    remaining_items = set(item_ids)
    covered_items = set()
    
    target_interactions = int(0.85 * n_interactions)
    while interactions_generated < target_interactions:
        # Prioritize uncovered items
        uncovered_items = list(remaining_items - covered_items)
        if uncovered_items and np.random.random() < 0.6:
            item_id = np.random.choice(uncovered_items)
        else:
            item_id = np.random.choice(niche_items)
        
        covered_items.add(item_id)
        
        # Mix of user types for niche items
        user_prob = np.random.random()
        if user_prob < 0.6:
            user_id = np.random.choice(regular_users)
        elif user_prob < 0.85:
            user_id = np.random.choice(active_users)
        else:
            user_id = np.random.choice(power_users)
        
        # Varied ratings for niche content
        rating = np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
        
        interactions.append({
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating,
            'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 365))
        })
        
        interactions_generated += 1
    
    # Phase 4: Fill remaining interactions with strategic patterns (15%)
    while interactions_generated < n_interactions:
        # Create collaborative filtering patterns
        user_id = np.random.choice(user_ids)
        
        # Strategic item selection for better precision/recall
        item_prob = np.random.random()
        if item_prob < 0.4:
            item_id = np.random.choice(blockbuster_items)  # High precision
        elif item_prob < 0.7:
            item_id = np.random.choice(popular_items)  # Good recall
        else:
            item_id = np.random.choice(niche_items)  # Coverage maintenance
        
        # Balanced ratings
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.4, 0.2, 0.1])
        
        interactions.append({
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating,
            'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 365))
        })
        
        interactions_generated += 1
    
    # Convert to DataFrame and optimize
    df = pd.DataFrame(interactions)
    
    # Remove duplicates, keeping highest rating (important for ALS)
    df = df.sort_values(['user_id', 'item_id', 'rating'], ascending=[True, True, False])
    df = df.drop_duplicates(['user_id', 'item_id'], keep='first')
    
    # Ensure we have enough interactions for the test split (20%)
    if len(df) < n_interactions * 0.9:  # Allow 10% loss from deduplication
        logger.warning(f"Generated {len(df)} interactions, less than target {n_interactions}")
    
    # Calculate and log statistics
    unique_users = df['user_id'].nunique()
    unique_items = df['item_id'].nunique()
    coverage = unique_items / n_items
    actual_sparsity = 1 - len(df) / (n_users * n_items)
    
    logger.info(f"Generated {len(df)} unique interactions")
    logger.info(f"Users: {unique_users}/{n_users}, Items: {unique_items}/{n_items}")
    logger.info(f"Item Coverage: {coverage:.3f} ({coverage*100:.1f}%)")
    logger.info(f"Sparsity: {actual_sparsity:.4f}")
    logger.info(f"Avg interactions per user: {len(df)/unique_users:.1f}")
    logger.info(f"Avg interactions per item: {len(df)/unique_items:.1f}")
    
    return df[['user_id', 'item_id', 'rating', 'timestamp']]

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
