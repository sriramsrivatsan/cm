import streamlit as st
import pandas as pd
from openai import OpenAI
import numpy as np
from pathlib import Path
import json
import os
from dotenv import load_dotenv
import traceback
import time
from sklearn.metrics.pairwise import cosine_similarity

# Try to import FAISS, fallback to sklearn if not available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    st.warning("FAISS not available. Using sklearn for similarity search as fallback.")

# Page configuration
st.set_page_config(
    page_title="Stateful Hybrid Query | Central Moment",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Central Moment branding
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Roboto:wght@300;400;500;700&display=swap');

    /* Global Styles - Central Moment Theme */
    .stApp {
        font-family: 'Inter', 'Roboto', sans-serif;
        background: linear-gradient(135deg, #0B4F6C 0%, #145DA0 50%, #2E8BC0 100%);
        min-height: 100vh;
    }

    /* Main Container */
    .main-header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(11, 79, 108, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #0B4F6C, #145DA0, #2E8BC0);
    }

    /* Logo and Title Styling */
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
        flex-wrap: wrap;
        gap: 1rem;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0B4F6C, #145DA0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin: 0;
        line-height: 1.2;
    }

    .subtitle {
        text-align: center;
        color: #0B4F6C;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        font-weight: 500;
    }

    .company-logo {
        max-height: 60px;
        max-width: 200px;
        object-fit: contain;
    }

    /* Status Cards */
    .status-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(11, 79, 108, 0.08);
        border-left: 4px solid #145DA0;
    }

    .status-success {
        border-left-color: #0C9;
        background: rgba(0, 204, 153, 0.05);
    }

    .status-error {
        border-left-color: #E74C3C;
        background: rgba(231, 76, 60, 0.05);
    }

    .status-warning {
        border-left-color: #F39C12;
        background: rgba(243, 156, 18, 0.05);
    }

    .status-info {
        border-left-color: #145DA0;
        background: rgba(20, 93, 160, 0.05);
    }

    /* Query Interface */
    .query-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 20px 40px rgba(11, 79, 108, 0.1);
        border: 1px solid rgba(20, 93, 160, 0.1);
    }

    /* Conversation History */
    .conversation-item {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #145DA0;
        box-shadow: 0 2px 10px rgba(11, 79, 108, 0.05);
    }

    .user-message {
        border-left-color: #0C9;
        background: rgba(0, 204, 153, 0.05);
    }

    .assistant-message {
        border-left-color: #145DA0;
        background: rgba(20, 93, 160, 0.05);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0B4F6C, #145DA0);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(20, 93, 160, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(20, 93, 160, 0.5);
        background: linear-gradient(135deg, #145DA0, #2E8BC0);
    }

    /* Sidebar */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
    }

    /* Data Display */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(11, 79, 108, 0.1);
        border: 1px solid rgba(20, 93, 160, 0.1);
    }

    /* Progress Indicators */
    .stProgress > div > div {
        background: linear-gradient(135deg, #0B4F6C, #145DA0);
    }

    /* Metrics */
    .metric-container {
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(11, 79, 108, 0.08);
        border: 1px solid rgba(20, 93, 160, 0.1);
    }

    /* Debug info styling */
    .debug-info {
        background: rgba(243, 156, 18, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 3px solid #F39C12;
        font-family: 'Roboto Mono', monospace;
        font-size: 0.85rem;
    }

    /* Footer Styling */
    .footer {
        background: linear-gradient(135deg, #0B4F6C, #145DA0);
        color: white;
        padding: 2.5rem 1rem;
        margin-top: 2rem;
        border-radius: 0;
        text-align: center;
        position: relative;
        width: 100%;
        box-sizing: border-box;
        clear: both;
        overflow: hidden;
    }

    .footer::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #2E8BC0, #145DA0, #0B4F6C);
    }

    .footer-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 1rem;
        position: relative;
        z-index: 1;
    }

    .footer-logo {
        max-height: 45px;
        margin-bottom: 1.5rem;
        opacity: 0.95;
        filter: brightness(1.2) contrast(1.1);
        display: block;
        margin-left: auto;
        margin-right: auto;
    }

    .footer-text {
        color: rgba(255, 255, 255, 0.95);
        line-height: 1.8;
        margin-bottom: 1.5rem;
        font-size: 1rem;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }

    .footer-links {
        display: flex;
        justify-content: center;
        gap: 2.5rem;
        margin: 1.5rem 0;
        flex-wrap: wrap;
    }

    .footer-links a {
        color: rgba(255, 255, 255, 0.85);
        text-decoration: none;
        transition: all 0.3s ease;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 500;
        border: 1px solid rgba(255, 255, 255, 0.2);
        background: rgba(255, 255, 255, 0.05);
    }

    .footer-links a:hover {
        color: white;
        background: rgba(255, 255, 255, 0.15);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }

    .copyright {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.8);
        border-top: 1px solid rgba(255, 255, 255, 0.2);
        padding-top: 1.5rem;
        margin-top: 1.5rem;
        line-height: 1.6;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }

    /* Ensure footer is always at bottom */
    .main > div {
        min-height: calc(100vh - 200px);
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }

        .logo-container {
            flex-direction: column;
        }

        .footer-links {
            flex-direction: column;
            gap: 1rem;
        }
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(20, 93, 160, 0.1);
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #0B4F6C, #145DA0);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #145DA0, #2E8BC0);
    }
</style>
""", unsafe_allow_html=True)

def display_status(message, status_type="info", icon="ℹ️"):
    """Display a status message with styling"""
    status_icons = {
        "success": "✅",
        "error": "❌",
        "warning": "⚠️",
        "info": "ℹ️"
    }

    icon = status_icons.get(status_type, icon)

    st.markdown(f"""
    <div class="status-card status-{status_type}">
        <strong>{icon} {message}</strong>
    </div>
    """, unsafe_allow_html=True)

def display_header():
    """Display the branded header with Central Moment logo and branding"""
    st.markdown("""
    <div class="main-header">
        <div class="logo-container">
            <img src="https://static.wixstatic.com/media/a9b51d_064c5beb866a4431b6f684ff26851545~mv2.png/v1/fill/w_198,h_42,al_c,q_85,usm_0.66_1.00_0.01,enc_avif,quality_auto/cmlogo_new_01.png"
                 alt="Central Moment Logo" class="company-logo">
            <div>
                <h1 class="main-title">📊 Stateful Hybrid Query</h1>
                <p class="subtitle">AI-powered data exploration with memory • We Measure Markets</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_footer():
    """Display the branded footer with Central Moment information using native Streamlit components"""
    # Add some spacing
    st.markdown("---")

    # Company logo and main info
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://static.wixstatic.com/media/a9b51d_064c5beb866a4431b6f684ff26851545~mv2.png/v1/fill/w_198,h_42,al_c,q_85,usm_0.66_1.00_0.01,enc_avif,quality_auto/cmlogo_new_01.png",
                width=200)
        st.markdown("**Central Moment Inc.** - *We Measure Markets*")
        st.caption("Transforming data solutions through market intelligence and sophisticated analytics")
        st.caption("Serving **100+ Countries** • **90+ Occupations** • **60+ Industries**")

    # Links section
    st.markdown("#### Quick Links")
    link_col1, link_col2, link_col3, link_col4, link_col5 = st.columns(5)

    with link_col1:
        st.markdown("[🌐 Website](https://www.centralmoment.com/)")
    with link_col2:
        st.markdown("[👥 About Us](https://www.centralmoment.com/aboutus)")
    with link_col3:
        st.markdown("[📧 Contact](mailto:info@centralmoment.com)")
    with link_col4:
        st.markdown("📊 Market Research")
    with link_col5:
        st.markdown("🔍 Data Integration")

    # Copyright section
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <strong>© 2024 Central Moment Inc. All rights reserved.</strong><br>
        "We Measure Markets" is a trademark of Central Moment Inc.<br>
        This application demonstrates advanced data analytics capabilities using AI-powered semantic search.<br>
        <em>Built with Streamlit, OpenAI GPT-4, and modern data science technologies.</em>
    </div>
    """, unsafe_allow_html=True)

def debug_file_search():
    """Debug function to show current directory and available files"""
    current_dir = Path.cwd()
    st.markdown(f"""
    <div class="debug-info">
        <strong>🔍 File Search Debug Info:</strong><br>
        📁 Current Directory: {current_dir}<br>
        📄 CSV files found: {list(current_dir.glob('*.csv'))}<br>
        📂 Directory contents (first 20): {list(current_dir.iterdir())[:20]}
    </div>
    """, unsafe_allow_html=True)

    return current_dir

def initialize_app():
    """Initialize the application with proper error handling"""
    initialization_steps = []

    try:
        # Step 1: Load environment variables
        with st.spinner("🔑 Loading environment variables..."):
            time.sleep(0.5)  # Visual feedback
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")

            if not api_key:
                initialization_steps.append(("❌ OpenAI API Key", "Not found in environment variables"))
                st.error("Please set your OPENAI_API_KEY in the .env file or environment variables")
                return None, None, None, initialization_steps
            else:
                initialization_steps.append(("✅ OpenAI API Key", "Successfully loaded"))

        # Step 2: Initialize OpenAI client
        with st.spinner("🤖 Initializing OpenAI client..."):
            time.sleep(0.5)
            try:
                client = OpenAI(api_key=api_key)
                # Test the connection with a simple call
                try:
                    client.models.list()
                    initialization_steps.append(("✅ OpenAI Client", "Successfully initialized and tested"))
                except Exception as test_e:
                    # If models.list() fails, just proceed (might be rate limited)
                    initialization_steps.append(("✅ OpenAI Client", "Initialized (connection test skipped)"))
                    st.info("OpenAI connection test skipped - proceeding with initialization")
            except Exception as e:
                initialization_steps.append(("❌ OpenAI Client", f"Failed to initialize: {str(e)}"))
                st.error(f"Failed to initialize OpenAI client: {str(e)}")
                return None, None, None, initialization_steps

        # Step 3: Load data with enhanced debugging
        with st.spinner("📄 Loading dataset..."):
            time.sleep(0.5)
            try:
                # Show debug information
                current_dir = debug_file_search()

                # Comprehensive search paths for different CSV file names
                possible_paths = [
                    "data.csv",
                    "data/data.csv",
                    "./data.csv",
                    "../data.csv",
                    "dataset.csv",
                    "market_data.csv",
                    "research_data.csv",
                    "Task Ratings (managers only).csv",
                    "data/Task Ratings (managers only).csv",
                    "../ONET data/ONET 26.3/Task Ratings (managers only).csv",
                    "../../ONET data/ONET 26.3/Task Ratings (managers only).csv"
                ]

                df = None
                used_path = None
                search_log = []

                st.info("🔍 Searching for CSV files...")
                for path in possible_paths:
                    try:
                        path_obj = Path(path)
                        exists = path_obj.exists()
                        search_log.append(f"{'✅' if exists else '❌'} {path} - {'Found' if exists else 'Not found'}")

                        if exists:
                            try:
                                df = pd.read_csv(path)
                                used_path = path
                                search_log.append(f"✅ Successfully loaded {path} with {len(df)} rows")
                                break
                            except Exception as load_error:
                                search_log.append(f"❌ Failed to load {path}: {str(load_error)}")
                    except Exception as check_error:
                        search_log.append(f"❌ Error checking {path}: {str(check_error)}")

                # Display search log
                st.markdown("### 🔍 File Search Log")
                for log_entry in search_log:
                    st.markdown(f"<div class='debug-info'>{log_entry}</div>", unsafe_allow_html=True)

                if df is None:
                    # Show file upload option
                    st.warning("CSV file not found in standard locations. You can upload a file or use sample data.")

                    uploaded_file = st.file_uploader(
                        "Upload your CSV file:",
                        type=['csv'],
                        help="Upload a CSV file with text data for analysis. Central Moment specializes in market research data."
                    )

                    if uploaded_file is not None:
                        try:
                            df = pd.read_csv(uploaded_file)
                            initialization_steps.append(("✅ Dataset", f"Loaded {len(df)} rows from uploaded file"))
                            st.success(f"Successfully loaded uploaded file with {len(df)} rows!")
                        except Exception as upload_error:
                            st.error(f"Failed to load uploaded file: {str(upload_error)}")
                            df = create_sample_dataset()
                            initialization_steps.append(("⚠️ Dataset", "Using sample data (upload failed)"))
                    else:
                        # If no file found and no upload, create a sample dataset
                        st.info("Using sample market research dataset for demonstration.")
                        df = create_sample_dataset()
                        initialization_steps.append(("⚠️ Dataset", "Using sample market research data"))
                else:
                    initialization_steps.append(("✅ Dataset", f"Loaded {len(df)} rows from {used_path}"))

            except Exception as e:
                initialization_steps.append(("❌ Dataset", f"Failed to load: {str(e)}"))
                st.error(f"Failed to load dataset: {str(e)}")
                st.code(traceback.format_exc())
                return None, None, None, initialization_steps

        # Step 4: Create embeddings and search index
        with st.spinner("🧠 Creating embeddings and search index..."):
            try:
                # Show available columns for debugging
                st.info(f"📊 Available columns: {list(df.columns)}")

                text_column = "Task"
                if text_column not in df.columns:
                    # Use the first text column available
                    text_columns = df.select_dtypes(include=['object']).columns
                    if len(text_columns) > 0:
                        text_column = text_columns[0]
                        st.warning(f"'Task' column not found. Using '{text_column}' instead.")
                    else:
                        raise ValueError("No text columns found in dataset")

                # Show sample data for verification
                st.info("📝 Sample data preview:")
                st.dataframe(df.head(3), use_container_width=True)

                # Create progress bar for embeddings
                progress_bar = st.progress(0)
                embeddings_list = []

                total_rows = min(len(df), 25)  # Slightly increased for better demo
                st.info(f"Creating embeddings for {total_rows} rows using OpenAI's advanced models...")

                for i, text in enumerate(df[text_column].iloc[:total_rows]):
                    try:
                        if pd.isna(text) or str(text).strip() == "":
                            st.warning(f"Empty text found at row {i}, using placeholder...")
                            embeddings_list.append(np.zeros(3072))
                        else:
                            embedding = get_embedding(str(text), client)
                            embeddings_list.append(embedding)
                        progress_bar.progress((i + 1) / total_rows)
                    except Exception as e:
                        st.warning(f"Failed to create embedding for row {i}: {str(e)}")
                        # Use zero vector as fallback
                        embeddings_list.append(np.zeros(3072))

                embeddings = np.vstack(embeddings_list)

                # Create search index (FAISS or sklearn fallback)
                if FAISS_AVAILABLE:
                    try:
                        index = faiss.IndexFlatL2(embeddings.shape[1])
                        index.add(embeddings.astype('float32'))
                        index_type = "FAISS"
                        initialization_steps.append(("✅ Search Index", f"FAISS index created with {len(embeddings)} embeddings"))
                    except Exception as faiss_e:
                        # Fallback to sklearn
                        index = embeddings  # Just store embeddings for sklearn
                        index_type = "sklearn"
                        initialization_steps.append(("✅ Search Index", f"Sklearn fallback index created with {len(embeddings)} embeddings"))
                else:
                    # Use sklearn as fallback
                    index = embeddings  # Just store embeddings for sklearn
                    index_type = "sklearn"
                    initialization_steps.append(("✅ Search Index", f"Sklearn index created with {len(embeddings)} embeddings"))

                # Store index type in session state for later use
                st.session_state.index_type = index_type

                # Limit df to the rows we processed
                df = df.iloc[:total_rows].copy()

            except Exception as e:
                initialization_steps.append(("❌ Search Index", f"Failed to create: {str(e)}"))
                st.error(f"Failed to create embeddings: {str(e)}")
                st.code(traceback.format_exc())
                return None, None, None, initialization_steps

        initialization_steps.append(("🎉 Initialization", "Complete! Ready to analyze market data."))
        return client, df, index, initialization_steps

    except Exception as e:
        initialization_steps.append(("❌ General Error", f"Unexpected error: {str(e)}"))
        st.error(f"Unexpected initialization error: {str(e)}")
        st.code(traceback.format_exc())
        return None, None, None, initialization_steps

def create_sample_dataset():
    """Create a sample market research dataset for demonstration"""
    sample_data = {
        'Market_Segment': ['Technology', 'Healthcare', 'Finance', 'Retail', 'Manufacturing',
                          'Education', 'Energy', 'Transportation', 'Real Estate', 'Media',
                          'Automotive', 'Telecommunications', 'Agriculture', 'Tourism', 'Construction'] * 2,
        'Region': ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East & Africa'] * 6,
        'Company_Size': ['Enterprise', 'Mid-Market', 'Small Business'] * 10,
        'Task': [
            'Analyze market trends and competitive landscape in technology sector',
            'Conduct primary research on healthcare service adoption rates',
            'Evaluate financial services market penetration strategies',
            'Assess retail customer behavior and purchasing patterns',
            'Study manufacturing supply chain optimization opportunities',
            'Research educational technology market sizing and demographics',
            'Analyze renewable energy market growth potential',
            'Evaluate transportation logistics and efficiency metrics',
            'Study real estate investment patterns and market dynamics',
            'Research media consumption trends across demographics',
            'Analyze automotive industry disruption and EV adoption',
            'Evaluate telecommunications infrastructure market needs',
            'Study agricultural technology adoption and market barriers',
            'Research tourism industry recovery and growth patterns',
            'Analyze construction market demand and pricing trends',
            'Conduct market sizing study for enterprise software solutions',
            'Evaluate healthcare digital transformation initiatives',
            'Research fintech adoption among small and medium businesses',
            'Study e-commerce market share and growth opportunities',
            'Analyze manufacturing automation and Industry 4.0 trends',
            'Research online education market expansion potential',
            'Evaluate energy storage market dynamics and forecasts',
            'Study smart transportation and mobility-as-a-service trends',
            'Analyze commercial real estate market post-pandemic shifts',
            'Research streaming media and content consumption patterns',
            'Evaluate electric vehicle charging infrastructure market',
            '5G telecommunications rollout impact assessment',
            'Study precision agriculture and smart farming adoption',
            'Research sustainable tourism and eco-travel preferences',
            'Analyze green building and sustainable construction trends'
        ],
        'Market_Value_Million': [1200, 850, 2100, 950, 1800, 450, 2200, 1100, 800, 650,
                               1950, 1400, 750, 420, 900, 1300, 890, 2050, 980, 1750,
                               480, 2180, 1080, 820, 670, 1920, 1380, 770, 440, 920],
        'Growth_Rate_Percent': [15.2, 8.7, 12.1, 6.9, 9.6, 11.3, 18.8, 7.4, 5.2, 10.4,
                              16.7, 13.2, 8.1, 4.8, 6.3, 14.8, 9.1, 11.9, 7.2, 8.9,
                              12.5, 17.6, 6.8, 5.9, 9.8, 15.9, 12.8, 7.7, 5.1, 6.7]
    }
    return pd.DataFrame(sample_data)

def get_embedding(text, client, model="text-embedding-3-large"):
    """Get embedding for text using OpenAI's advanced embedding model"""
    text = str(text).replace("\n", " ")
    try:
        resp = client.embeddings.create(input=[text], model=model)
        return np.array(resp.data[0].embedding)
    except Exception as e:
        st.warning(f"Embedding API call failed: {str(e)}")
        # Return zero vector as fallback
        return np.zeros(3072)

def similarity_search(query_embedding, index, k=None):
    """Perform similarity search using either FAISS or sklearn"""
    if k is None:
        k = len(index) if not FAISS_AVAILABLE else index.ntotal

    if FAISS_AVAILABLE and hasattr(index, 'search'):
        # Use FAISS
        try:
            _, indices = index.search(np.array([query_embedding]), k)
            return indices[0].tolist()
        except Exception as e:
            st.warning(f"FAISS search failed, falling back to sklearn: {str(e)}")
            # Fallback to sklearn
            similarities = cosine_similarity([query_embedding], index)[0]
            indices = np.argsort(similarities)[::-1][:k]
            return indices.tolist()
    else:
        # Use sklearn cosine similarity
        similarities = cosine_similarity([query_embedding], index)[0]
        indices = np.argsort(similarities)[::-1][:k]
        return indices.tolist()

def query_data(user_query, client, df, index, conversation_history, current_filters):
    """Process user query with memory/state - Central Moment market research focus"""
    try:
        conversation_history.append({"role": "user", "content": user_query})

        state_summary = f"Current filters: {current_filters}"

        system_prompt = (
            "You are a Central Moment market research data assistant specializing in market intelligence and analytics.\n"
            "You are working with a pandas DataFrame containing market research data with text columns and structured market information.\n"
            "You have memory of previous filters and can build upon past queries.\n"
            f"{state_summary}\n"
            "Given the following query, decide which structured or semantic filters to apply, and return them as a JSON object.\n"
            "For semantic queries about market trends, industries, or research topics, use the key 'semantic_query' with the search terms.\n"
            "For structured filters like market segments, regions, or company sizes, use appropriate column names as keys.\n"
            "Focus on market research, business intelligence, and data analysis terminology.\n"
            "Respond only with valid JSON."
        )

        messages = [{"role": "system", "content": system_prompt}] + conversation_history

        with st.spinner("🤖 Processing your market research query..."):
            resp = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0
            )

        gpt_reply = resp.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": gpt_reply})

        # Display GPT response
        with st.expander("🤖 GPT-4 Market Analysis", expanded=False):
            st.code(gpt_reply, language="json")

        try:
            new_filters = json.loads(gpt_reply)
            current_filters.update(new_filters)
        except json.JSONDecodeError:
            st.warning("Could not parse GPT response as JSON. Using semantic search instead.")
            new_filters = {"semantic_query": user_query}
            current_filters.update(new_filters)

        # Apply filters
        structured_mask = pd.Series([True] * len(df))

        # Apply semantic filtering
        semantic_mask = pd.Series([True] * len(df))
        if "semantic_query" in current_filters:
            try:
                query_emb = get_embedding(current_filters["semantic_query"], client).astype("float32")
                matched_indices = similarity_search(query_emb, index, len(df))

                # Take top 60% most similar results for semantic filtering (more inclusive for market research)
                top_k = max(1, int(len(matched_indices) * 0.6))
                top_indices = matched_indices[:top_k]
                semantic_mask = df.index.isin(top_indices)

            except Exception as e:
                st.warning(f"Semantic search failed: {str(e)}")
                st.code(traceback.format_exc())

        # Apply other structured filters
        for key, value in current_filters.items():
            if key != "semantic_query" and key in df.columns:
                try:
                    if isinstance(value, dict) and "contains" in value:
                        if isinstance(value["contains"], list):
                            mask = df[key].astype(str).str.contains('|'.join(value["contains"]), case=False, na=False)
                        else:
                            mask = df[key].astype(str).str.contains(str(value["contains"]), case=False, na=False)
                        structured_mask = structured_mask & mask
                    else:
                        structured_mask = structured_mask & (df[key] == value)
                except Exception as e:
                    st.warning(f"Could not apply filter {key}={value}: {str(e)}")

        final_df = df[structured_mask & semantic_mask]

        return final_df, conversation_history, current_filters

    except Exception as e:
        st.error(f"Query processing failed: {str(e)}")
        st.code(traceback.format_exc())
        return df, conversation_history, current_filters

def main():
    # Display branded header
    display_header()

    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.conversation_history = []
        st.session_state.current_filters = {}
        st.session_state.client = None
        st.session_state.df = None
        st.session_state.index = None
        st.session_state.initialization_steps = []

    # Sidebar for initialization status
    with st.sidebar:
        st.markdown("### 🚀 System Status")
        st.markdown("*Central Moment Market Intelligence Platform*")

        if not st.session_state.initialized:
            if st.button("🔄 Initialize Platform", use_container_width=True):
                with st.container():
                    client, df, index, steps = initialize_app()
                    st.session_state.initialization_steps = steps

                    if client is not None:
                        st.session_state.client = client
                        st.session_state.df = df
                        st.session_state.index = index
                        st.session_state.initialized = True
                        st.rerun()

        # Display initialization steps
        for step, message in st.session_state.initialization_steps:
            if "✅" in step:
                display_status(f"{step}: {message}", "success")
            elif "❌" in step:
                display_status(f"{step}: {message}", "error")
            elif "⚠️" in step:
                display_status(f"{step}: {message}", "warning")
            else:
                display_status(f"{step}: {message}", "info")

        if st.session_state.initialized:
            st.markdown("### 📊 Market Data Metrics")
            if st.session_state.df is not None:
                st.metric("Total Records", len(st.session_state.df))
                st.metric("Active Filters", len(st.session_state.current_filters))
                st.metric("Query Sessions", len(st.session_state.conversation_history) // 2)

                # Additional market-specific metrics
                if 'Market_Value_Million' in st.session_state.df.columns:
                    total_market_value = st.session_state.df['Market_Value_Million'].sum()
                    st.metric("Total Market Value", f"${total_market_value:,.0f}M")

                if 'Growth_Rate_Percent' in st.session_state.df.columns:
                    avg_growth = st.session_state.df['Growth_Rate_Percent'].mean()
                    st.metric("Avg. Growth Rate", f"{avg_growth:.1f}%")

    # Main interface
    if not st.session_state.initialized:
        # Use Streamlit native components instead of HTML for better compatibility
        st.markdown("## 🔧 Platform Setup Required")
        st.info("Please initialize the Central Moment market intelligence platform using the sidebar to get started.")

        # About section using Streamlit expander
        with st.expander("🏢 About Central Moment", expanded=True):
            st.markdown("""
            **Central Moment Inc.** specializes in market intelligence and data integration across:

            - **100+ Countries** - Global market coverage
            - **90+ Occupations** - Comprehensive workforce analysis
            - **60+ Industries** - Cross-sector market research
            - **9+ Enterprise Sizes** - From startups to Fortune 500
            """)

        # Requirements section
        st.markdown("### System Requirements:")
        st.markdown("""
        - OpenAI API key for advanced AI analytics
        - Market research dataset (CSV format) - auto-detected or uploadable
        - Internet connection for real-time API calls
        """)

        # Data sources section
        st.markdown("### Supported Data Sources:")
        st.markdown("""
        - Market sizing and segmentation data
        - Industry analysis and competitive intelligence
        - Consumer behavior and preference studies
        - Demographic and psychographic profiles
        """)

        # Display footer for uninitialized state
        display_footer()
        return

    # Query interface (only shown when initialized)
    st.markdown("### 💬 Market Research Query Interface")
    st.caption("*Ask questions about market trends, industry analysis, competitive intelligence, and more...*")
    st.markdown("### 💬 Market Research Query Interface")
    st.markdown("*Ask questions about market trends, industry analysis, competitive intelligence, and more...*")

    col1, col2 = st.columns([4, 1])

    with col1:
        user_query = st.text_input(
            "Enter your market research query:",
            placeholder="e.g., Show me technology sector growth trends, or Find healthcare market opportunities",
            label_visibility="collapsed"
        )

    with col2:
        query_submitted = st.button("🔍 Analyze", use_container_width=True)

    # Quick query suggestions
    st.markdown("**Quick Queries:** " + " | ".join([
        "`technology trends`",
        "`healthcare market size`",
        "`enterprise segments`",
        "`high growth markets`",
        "`regional analysis`"
    ]))

    # Process query
    if query_submitted and user_query:
        result_df, conversation_history, current_filters = query_data(
            user_query,
            st.session_state.client,
            st.session_state.df,
            st.session_state.index,
            st.session_state.conversation_history,
            st.session_state.current_filters
        )

        # Update session state
        st.session_state.conversation_history = conversation_history
        st.session_state.current_filters = current_filters

        # Display results
        if len(result_df) > 0:
            st.success(f"Found {len(result_df)} matching market research records")

            # Market insights summary
            col1, col2, col3 = st.columns(3)

            with col1:
                if 'Market_Value_Million' in result_df.columns:
                    total_value = result_df['Market_Value_Million'].sum()
                    st.metric("Total Market Value", f"${total_value:,.0f}M")

            with col2:
                if 'Growth_Rate_Percent' in result_df.columns:
                    avg_growth = result_df['Growth_Rate_Percent'].mean()
                    st.metric("Average Growth Rate", f"{avg_growth:.1f}%")

            with col3:
                unique_segments = result_df['Market_Segment'].nunique() if 'Market_Segment' in result_df.columns else 0
                st.metric("Market Segments", unique_segments)

            # Display current filters
            if st.session_state.current_filters:
                st.markdown("### 🔧 Active Market Filters")
                st.json(st.session_state.current_filters)

            # Display results
            st.markdown("### 📊 Market Research Results")
            st.dataframe(result_df, use_container_width=True, height=400)

            # Download button
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="📥 Export Market Data (CSV)",
                data=csv,
                file_name=f"central_moment_market_analysis_{len(result_df)}_records.csv",
                mime="text/csv",
                help="Download filtered market research data for further analysis"
            )
        else:
            st.warning("No market research records found matching your query. Try adjusting your search terms or broadening your criteria.")

    # Conversation history
    if st.session_state.conversation_history:
        st.markdown("### 💭 Analysis History")
        st.caption("*Track your market research queries and build upon previous insights*")

        for i in range(0, len(st.session_state.conversation_history), 2):
            if i + 1 < len(st.session_state.conversation_history):
                user_msg = st.session_state.conversation_history[i]["content"]
                assistant_msg = st.session_state.conversation_history[i + 1]["content"]

                # Use Streamlit chat elements instead of HTML
                with st.container():
                    st.markdown(f"**🧑 Market Analyst:** {user_msg}")
                    with st.expander("🤖 Central Moment AI Response"):
                        st.code(assistant_msg, language="json")

    # Reset button
    if st.session_state.conversation_history or st.session_state.current_filters:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset Analysis Session"):
                st.session_state.conversation_history = []
                st.session_state.current_filters = {}
                st.rerun()
        with col2:
            if st.button("📊 New Market Study"):
                st.session_state.conversation_history = []
                st.session_state.current_filters = {}
                st.session_state.initialized = False
                st.rerun()

    # Always display footer at the end
    display_footer()

if __name__ == "__main__":
    main()
