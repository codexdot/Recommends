import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta

from data_processor import DataProcessor
from recommender import ImplicitRecommender
from evaluator import RecommendationEvaluator
from cold_start_handler import ColdStartHandler
from utils import generate_sample_data, format_recommendations

st.set_page_config(
    page_title="Implicit Feedback Recommender System",
    page_icon="🎯",
    layout="wide"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'evaluation_complete' not in st.session_state:
    st.session_state.evaluation_complete = False

def main():
    st.title("🎯 Implicit Feedback Recommendation System")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section",
        ["Data Overview", "Model Training", "Generate Recommendations", "Evaluation", "Performance Metrics"]
    )
    
    if page == "Data Overview":
        data_overview_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Generate Recommendations":
        recommendations_page()
    elif page == "Evaluation":
        evaluation_page()
    elif page == "Performance Metrics":
        metrics_page()

def data_overview_page():
    st.header("📊 Data Overview")
    
    # Data loading section
    st.subheader("Data Loading")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        data_source = st.selectbox(
            "Select data source:",
            ["Generate Sample Data", "Upload CSV File"]
        )
    
    with col2:
        if st.button("Load Data", type="primary"):
            with st.spinner("Loading data..."):
                if data_source == "Generate Sample Data":
                    # Generate synthetic implicit feedback data
                    interactions_df = generate_sample_data()
                    st.session_state.interactions_df = interactions_df
                    st.session_state.data_loaded = True
                    st.success("Sample data generated successfully!")
                else:
                    st.info("Please upload a CSV file with columns: user_id, item_id, rating (implicit feedback score)")
    
    # File upload for custom data
    if data_source == "Upload CSV File":
        uploaded_file = st.file_uploader(
            "Upload your implicit feedback data (CSV)",
            type=['csv'],
            help="Expected format: user_id, item_id, rating"
        )
        
        if uploaded_file is not None:
            try:
                interactions_df = pd.read_csv(uploaded_file)
                required_columns = ['user_id', 'item_id', 'rating']
                
                if all(col in interactions_df.columns for col in required_columns):
                    st.session_state.interactions_df = interactions_df
                    st.session_state.data_loaded = True
                    st.success("Data uploaded successfully!")
                else:
                    st.error(f"Please ensure your CSV has columns: {required_columns}")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    # Display data overview if loaded
    if st.session_state.data_loaded:
        df = st.session_state.interactions_df
        
        st.subheader("Dataset Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Interactions", f"{len(df):,}")
        with col2:
            st.metric("Unique Users", f"{len(df['user_id'].unique()):,}")
        with col3:
            st.metric("Unique Items", f"{len(df['item_id'].unique()):,}")
        with col4:
            sparsity = 1 - (len(df) / (len(df['user_id'].unique()) * len(df['item_id'].unique())))
            st.metric("Sparsity", f"{sparsity:.2%}")
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Visualizations
        st.subheader("Data Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # User interaction distribution
            user_counts = df['user_id'].value_counts().head(20)
            fig_users = px.bar(
                x=user_counts.index.astype(str),
                y=user_counts.values,
                title="Top 20 Users by Interaction Count",
                labels={'x': 'User ID', 'y': 'Interaction Count'}
            )
            st.plotly_chart(fig_users, use_container_width=True)
        
        with col2:
            # Item popularity distribution
            item_counts = df['item_id'].value_counts().head(20)
            fig_items = px.bar(
                x=item_counts.index.astype(str),
                y=item_counts.values,
                title="Top 20 Most Popular Items",
                labels={'x': 'Item ID', 'y': 'Interaction Count'}
            )
            st.plotly_chart(fig_items, use_container_width=True)
        
        # Rating distribution
        fig_ratings = px.histogram(
            df, x='rating',
            title="Distribution of Implicit Feedback Scores",
            labels={'rating': 'Implicit Feedback Score', 'count': 'Frequency'}
        )
        st.plotly_chart(fig_ratings, use_container_width=True)

def model_training_page():
    st.header("🚀 Model Training")
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Data Overview page.")
        return
    
    st.subheader("Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        algorithm = st.selectbox(
            "Select Algorithm:",
            ["Alternating Least Squares (ALS)", "Bayesian Personalized Ranking (BPR)", "Logistic Matrix Factorization (LMF)"]
        )
        
        factors = st.slider("Number of Factors", min_value=10, max_value=200, value=50, step=10)
        regularization = st.slider("Regularization", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    
    with col2:
        iterations = st.slider("Iterations", min_value=10, max_value=100, value=20, step=5)
        alpha = st.slider("Alpha (Confidence)", min_value=1, max_value=50, value=15, step=1)
        test_size = st.slider("Test Split Size", min_value=0.1, max_value=0.3, value=0.2, step=0.05)
    
    if st.button("Train Model", type="primary"):
        with st.spinner("Training recommendation model..."):
            try:
                # Initialize data processor
                processor = DataProcessor()
                
                # Process data
                user_item_matrix, user_mapping, item_mapping = processor.create_user_item_matrix(
                    st.session_state.interactions_df
                )
                
                # Split data
                train_matrix, test_matrix = processor.train_test_split(
                    user_item_matrix, test_size=test_size
                )
                
                # Map algorithm display names to internal names
                algorithm_mapping = {
                    "Alternating Least Squares (ALS)": "als",
                    "Bayesian Personalized Ranking (BPR)": "bpr", 
                    "Logistic Matrix Factorization (LMF)": "lmf"
                }
                
                # Initialize recommender
                recommender = ImplicitRecommender(
                    algorithm=algorithm_mapping.get(algorithm, "als"),
                    factors=factors,
                    regularization=regularization,
                    iterations=iterations,
                    alpha=alpha
                )
                
                # Train model
                start_time = time.time()
                recommender.fit(train_matrix)
                training_time = time.time() - start_time
                
                # Train cold start handler
                cold_start_handler = ColdStartHandler()
                cold_start_handler.fit(train_matrix, user_mapping, item_mapping)
                
                # Store in session state
                st.session_state.recommender = recommender
                st.session_state.cold_start_handler = cold_start_handler
                st.session_state.user_item_matrix = user_item_matrix
                st.session_state.train_matrix = train_matrix
                st.session_state.test_matrix = test_matrix
                st.session_state.user_mapping = user_mapping
                st.session_state.item_mapping = item_mapping
                st.session_state.model_trained = True
                st.session_state.training_time = training_time
                
                st.success(f"Model trained successfully in {training_time:.2f} seconds!")
                
                # Display training metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Training Time", f"{training_time:.2f}s")
                with col2:
                    st.metric("Algorithm", algorithm_mapping.get(algorithm, "ALS").upper())
                with col3:
                    st.metric("Factors", factors)
                
            except Exception as e:
                st.error(f"Error during training: {str(e)}")

def recommendations_page():
    st.header("🎯 Generate Recommendations")
    
    if not st.session_state.model_trained:
        st.warning("Please train a model first from the Model Training page.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Recommendation Settings")
        
        # User selection
        available_users = list(st.session_state.user_mapping.keys())
        user_options = ["New User (Cold Start)"] + available_users
        selected_user_option = st.selectbox("Select User:", user_options)
        
        num_recommendations = st.slider("Number of Recommendations:", 1, 20, 10)
        
        include_explanations = st.checkbox("Include Explanations", value=True)
        
        # Cold start strategy selection
        cold_start_strategy = "Hybrid"  # Default value
        if selected_user_option == "New User (Cold Start)":
            st.subheader("Cold Start Strategy")
            cold_start_strategy = st.selectbox(
                "Recommendation Strategy:",
                ["Hybrid", "Popular Items", "Diverse Selection", "Trending Items"]
            )
        
        if st.button("Generate Recommendations", type="primary"):
            with st.spinner("Generating recommendations..."):
                try:
                    if selected_user_option == "New User (Cold Start)":
                        # Use cold start handler
                        strategy_map = {
                            "Hybrid": "hybrid",
                            "Popular Items": "popular", 
                            "Diverse Selection": "diverse",
                            "Trending Items": "trending"
                        }
                        
                        recommendations = st.session_state.cold_start_handler.recommend_for_cold_start_user(
                            n_recommendations=num_recommendations,
                            strategy=strategy_map.get(cold_start_strategy, "hybrid")
                        )
                        
                        st.session_state.current_recommendations = recommendations
                        st.session_state.current_user = "New User (Cold Start)"
                    else:
                        # Regular user recommendations
                        recommendations = st.session_state.recommender.recommend_for_user(
                            user_id=selected_user_option,
                            user_item_matrix=st.session_state.user_item_matrix,
                            user_mapping=st.session_state.user_mapping,
                            item_mapping=st.session_state.item_mapping,
                            n_recommendations=num_recommendations,
                            include_explanations=include_explanations
                        )
                        
                        st.session_state.current_recommendations = recommendations
                        st.session_state.current_user = selected_user_option
                    
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")
    
    with col2:
        st.subheader("Recommendations")
        
        if 'current_recommendations' in st.session_state:
            recommendations = st.session_state.current_recommendations
            user_id = st.session_state.current_user
            
            st.write(f"**Recommendations for User {user_id}:**")
            
            # Display recommendations in enhanced format
            for i, (item_id, score, explanation) in enumerate(recommendations, 1):
                with st.container():
                    # Create an attractive card-like layout
                    st.markdown(f"""
                    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 10px 0; background-color: #f9f9f9;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h4 style="margin: 0; color: #1f77b4;">#{i} Item {item_id}</h4>
                            <span style="background-color: #1f77b4; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px;">
                                Score: {score:.3f}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if include_explanations and explanation:
                        st.markdown(f"💡 **Why this recommendation?**")
                        st.markdown(f"_{explanation}_")
                    
            
            # Recommendation insights and user history
            st.subheader("📊 Recommendation Insights")
            
            # Calculate recommendation statistics
            scores = [score for _, score, _ in recommendations]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_score = np.mean(scores) if scores else 0
                st.metric("Average Confidence", f"{avg_score:.3f}")
                
            with col2:
                score_range = max(scores) - min(scores) if len(scores) > 1 else 0
                st.metric("Score Range", f"{score_range:.3f}")
                
            with col3:
                recommendation_type = "Cold Start" if user_id == "New User (Cold Start)" else "Personalized"
                st.metric("Recommendation Type", recommendation_type)
            
            # Show user's interaction history
            if user_id != "New User (Cold Start)":
                st.subheader("📈 User's Interaction History")
                user_history = st.session_state.interactions_df[
                    st.session_state.interactions_df['user_id'] == user_id
                ].sort_values('rating', ascending=False)
                
                if not user_history.empty:
                    st.dataframe(user_history.head(10), use_container_width=True)
                    st.caption(f"Showing top 10 of {len(user_history)} total interactions")
                else:
                    st.info("This user has no interaction history")
            else:
                st.subheader("🆕 Cold Start Information")
                st.info("These recommendations are designed for new users without interaction history. They combine popular, diverse, and trending items to help you discover content you might enjoy.")
                
                if 'cold_start_handler' in st.session_state:
                    stats = st.session_state.cold_start_handler.get_cold_start_statistics()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Available Popular Items", stats['popular_items_count'])
                    with col2:
                        st.metric("Item Categories", stats['item_clusters'])

def evaluation_page():
    st.header("📈 Model Evaluation")
    
    if not st.session_state.model_trained:
        st.warning("Please train a model first from the Model Training page.")
        return
    
    if st.button("Run Evaluation", type="primary"):
        with st.spinner("Evaluating model performance..."):
            try:
                # Initialize evaluator
                evaluator = RecommendationEvaluator()
                
                # Run evaluation
                evaluation_results = evaluator.evaluate_model(
                    recommender=st.session_state.recommender,
                    train_matrix=st.session_state.train_matrix,
                    test_matrix=st.session_state.test_matrix,
                    user_mapping=st.session_state.user_mapping,
                    item_mapping=st.session_state.item_mapping,
                    k_values=[5, 10, 20]
                )
                
                st.session_state.evaluation_results = evaluation_results
                st.session_state.evaluation_complete = True
                
                st.success("Evaluation completed successfully!")
                
            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")
    
    if st.session_state.evaluation_complete:
        results = st.session_state.evaluation_results
        
        st.subheader("Evaluation Metrics")
        
        # Display metrics in tabs
        tab1, tab2, tab3 = st.tabs(["Precision & Recall", "NDCG", "Coverage"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Precision@K**")
                precision_data = []
                for k in [5, 10, 20]:
                    precision_data.append({
                        'K': k,
                        'Precision': float(results[f'precision_at_{k}'])
                    })
                
                df_precision = pd.DataFrame(precision_data)
                fig_precision = px.bar(df_precision, x='K', y='Precision', 
                                     title="Precision@K")
                st.plotly_chart(fig_precision, use_container_width=True)
            
            with col2:
                st.write("**Recall@K**")
                recall_data = []
                for k in [5, 10, 20]:
                    recall_data.append({
                        'K': k,
                        'Recall': float(results[f'recall_at_{k}'])
                    })
                
                df_recall = pd.DataFrame(recall_data)
                fig_recall = px.bar(df_recall, x='K', y='Recall', 
                                  title="Recall@K")
                st.plotly_chart(fig_recall, use_container_width=True)
        
        with tab2:
            st.write("**NDCG@K**")
            ndcg_data = []
            for k in [5, 10, 20]:
                ndcg_data.append({
                    'K': k,
                    'NDCG': float(results[f'ndcg_at_{k}'])
                })
            
            df_ndcg = pd.DataFrame(ndcg_data)
            fig_ndcg = px.bar(df_ndcg, x='K', y='NDCG', 
                             title="NDCG@K")
            st.plotly_chart(fig_ndcg, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Item Coverage", f"{results['coverage']:.2%}")
                st.caption("Percentage of items that appear in at least one recommendation")
            
            with col2:
                st.metric("Intra-list Diversity", f"{results['diversity']:.3f}")
                st.caption("Average diversity within recommendation lists")

def metrics_page():
    st.header("⚡ Performance Metrics")
    
    if not st.session_state.model_trained:
        st.warning("Please train a model first from the Model Training page.")
        return
    
    # Performance benchmarks
    st.subheader("Training Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'training_time' in st.session_state:
            st.metric("Training Time", f"{st.session_state.training_time:.2f}s")
    
    with col2:
        # Calculate model size
        model_size = st.session_state.recommender.get_model_size()
        st.metric("Model Size", f"{model_size:.2f} MB")
    
    with col3:
        # Memory usage estimation
        matrix_size = st.session_state.user_item_matrix.data.nbytes / (1024**2)
        st.metric("Matrix Size", f"{matrix_size:.2f} MB")
    
    # Scalability analysis
    st.subheader("Scalability Analysis")
    
    # Simulate performance with different data sizes
    data_sizes = [1000, 5000, 10000, 50000, 100000]
    estimated_times = []
    
    current_interactions = len(st.session_state.interactions_df)
    current_time = st.session_state.training_time if 'training_time' in st.session_state else 1.0
    
    for size in data_sizes:
        # Simple linear scaling estimation
        estimated_time = (size / current_interactions) * current_time
        estimated_times.append(estimated_time)
    
    scalability_df = pd.DataFrame({
        'Interactions': data_sizes,
        'Estimated Training Time (s)': estimated_times
    })
    
    fig_scalability = px.line(scalability_df, x='Interactions', y='Estimated Training Time (s)',
                             title="Scalability: Training Time vs Dataset Size")
    st.plotly_chart(fig_scalability, use_container_width=True)
    
    # Cold start performance
    st.subheader("Cold Start Analysis")
    
    if 'cold_start_handler' in st.session_state:
        # Display cold start handler statistics
        cold_start_stats = st.session_state.cold_start_handler.get_cold_start_statistics()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Popular Items Pool", f"{cold_start_stats['popular_items_count']:,}")
            st.caption("Items for popularity-based recommendations")
        
        with col2:
            st.metric("Diverse Items Pool", f"{cold_start_stats['diverse_items_count']:,}")
            st.caption("Items for diversity-based recommendations")
        
        with col3:
            st.metric("Trending Items Pool", f"{cold_start_stats['trending_items_count']:,}")
            st.caption("Items for trending recommendations")
        
        col4, col5 = st.columns(2)
        
        with col4:
            st.metric("User Clusters", f"{cold_start_stats['user_clusters']:,}")
            st.caption("User behavior clusters")
        
        with col5:
            st.metric("Item Clusters", f"{cold_start_stats['item_clusters']:,}")
            st.caption("Item similarity clusters")
    
    if st.session_state.evaluation_complete:
        # Analyze cold start users (users with few interactions)
        interactions_per_user = st.session_state.interactions_df['user_id'].value_counts()
        cold_start_users = interactions_per_user[interactions_per_user <= 5].index.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Cold Start Users", f"{len(cold_start_users):,}")
            st.caption("Users with ≤5 interactions")
        
        with col2:
            cold_start_percentage = len(cold_start_users) / len(interactions_per_user) * 100
            st.metric("Cold Start Percentage", f"{cold_start_percentage:.1f}%")
    
    # Model configuration summary
    st.subheader("Model Configuration")
    
    config_data = {
        'Parameter': ['Algorithm', 'Factors', 'Regularization', 'Iterations', 'Alpha'],
        'Value': [
            str(st.session_state.recommender.algorithm),
            str(st.session_state.recommender.factors),
            str(st.session_state.recommender.regularization),
            str(st.session_state.recommender.iterations),
            str(st.session_state.recommender.alpha)
        ]
    }
    
    config_df = pd.DataFrame(config_data)
    st.dataframe(config_df, use_container_width=True)

if __name__ == "__main__":
    main()
