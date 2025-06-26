import pickle
import json
import os
from datetime import datetime
import zipfile
import tempfile
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Handles saving and loading complete recommendation model packages.
    """
    
    def __init__(self):
        self.model_directory = "saved_models"
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)
    
    def save_complete_model(self, recommender, user_mapping, item_mapping, 
                           user_item_matrix, cold_start_handler=None, 
                           evaluation_results=None, filename=None):
        """
        Save a complete model package including all necessary components.
        
        Args:
            recommender: Trained recommendation model
            user_mapping: User ID to index mapping
            item_mapping: Item ID to index mapping
            user_item_matrix: User-item interaction matrix
            cold_start_handler: Cold start handler (optional)
            evaluation_results: Model evaluation results (optional)
            filename: Custom filename (optional)
            
        Returns:
            str: Path to saved model package
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recommendation_model_package_{timestamp}.zip"
        
        if not filename.endswith('.zip'):
            filename += '.zip'
        
        filepath = os.path.join(self.model_directory, filename)
        
        # Create temporary directory for model components
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save main model
            model_path = os.path.join(temp_dir, "model.pkl")
            recommender.save_model(model_path)
            
            # Save mappings
            mappings_data = {
                'user_mapping': user_mapping,
                'item_mapping': item_mapping,
                'matrix_shape': user_item_matrix.shape,
                'matrix_nnz': user_item_matrix.nnz
            }
            mappings_path = os.path.join(temp_dir, "mappings.pkl")
            with open(mappings_path, 'wb') as f:
                pickle.dump(mappings_data, f)
            
            # Save matrix (sparse format)
            matrix_path = os.path.join(temp_dir, "user_item_matrix.pkl")
            with open(matrix_path, 'wb') as f:
                pickle.dump(user_item_matrix, f)
            
            # Save cold start handler if available
            if cold_start_handler is not None:
                cold_start_path = os.path.join(temp_dir, "cold_start_handler.pkl")
                with open(cold_start_path, 'wb') as f:
                    pickle.dump(cold_start_handler, f)
            
            # Save evaluation results if available
            if evaluation_results is not None:
                eval_path = os.path.join(temp_dir, "evaluation_results.json")
                with open(eval_path, 'w') as f:
                    json.dump(evaluation_results, f, indent=2)
            
            # Create model metadata
            metadata = {
                'package_version': '1.0',
                'creation_timestamp': datetime.now().isoformat(),
                'model_algorithm': recommender.algorithm,
                'model_factors': recommender.factors,
                'n_users': len(user_mapping),
                'n_items': len(item_mapping),
                'matrix_sparsity': 1 - (user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1])),
                'has_cold_start_handler': cold_start_handler is not None,
                'has_evaluation_results': evaluation_results is not None
            }
            
            metadata_path = os.path.join(temp_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create ZIP package
            with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(model_path, "model.pkl")
                zipf.write(mappings_path, "mappings.pkl")
                zipf.write(matrix_path, "user_item_matrix.pkl")
                zipf.write(metadata_path, "metadata.json")
                
                if cold_start_handler is not None:
                    zipf.write(cold_start_path, "cold_start_handler.pkl")
                
                if evaluation_results is not None:
                    zipf.write(eval_path, "evaluation_results.json")
        
        logger.info(f"Complete model package saved to {filepath}")
        return filepath
    
    def load_complete_model(self, filepath):
        """
        Load a complete model package.
        
        Args:
            filepath: Path to the model package ZIP file
            
        Returns:
            dict: Dictionary containing all model components
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model package not found: {filepath}")
        
        # Create temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract ZIP file
            with zipfile.ZipFile(filepath, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            # Load metadata
            metadata_path = os.path.join(temp_dir, "metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load main model
            from recommender import ImplicitRecommender
            recommender = ImplicitRecommender()
            model_path = os.path.join(temp_dir, "model.pkl")
            recommender.load_model(model_path)
            
            # Load mappings
            mappings_path = os.path.join(temp_dir, "mappings.pkl")
            with open(mappings_path, 'rb') as f:
                mappings_data = pickle.load(f)
            
            # Load matrix
            matrix_path = os.path.join(temp_dir, "user_item_matrix.pkl")
            with open(matrix_path, 'rb') as f:
                user_item_matrix = pickle.load(f)
            
            # Load cold start handler if available
            cold_start_handler = None
            cold_start_path = os.path.join(temp_dir, "cold_start_handler.pkl")
            if os.path.exists(cold_start_path):
                with open(cold_start_path, 'rb') as f:
                    cold_start_handler = pickle.load(f)
            
            # Load evaluation results if available
            evaluation_results = None
            eval_path = os.path.join(temp_dir, "evaluation_results.json")
            if os.path.exists(eval_path):
                with open(eval_path, 'r') as f:
                    evaluation_results = json.load(f)
        
        logger.info(f"Complete model package loaded from {filepath}")
        
        return {
            'recommender': recommender,
            'user_mapping': mappings_data['user_mapping'],
            'item_mapping': mappings_data['item_mapping'],
            'user_item_matrix': user_item_matrix,
            'cold_start_handler': cold_start_handler,
            'evaluation_results': evaluation_results,
            'metadata': metadata
        }
    
    def list_saved_models(self):
        """
        List all saved model packages.
        
        Returns:
            list: List of model package filenames
        """
        if not os.path.exists(self.model_directory):
            return []
        
        model_files = [f for f in os.listdir(self.model_directory) if f.endswith('.zip')]
        return sorted(model_files, reverse=True)  # Most recent first
    
    def get_model_info(self, filename):
        """
        Get metadata information for a saved model package.
        
        Args:
            filename: Name of the model package file
            
        Returns:
            dict: Model metadata
        """
        filepath = os.path.join(self.model_directory, filename)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            with zipfile.ZipFile(filepath, 'r') as zipf:
                with zipf.open("metadata.json") as f:
                    metadata = json.load(f)
            return metadata
        except Exception as e:
            logger.error(f"Error reading model metadata: {e}")
            return None