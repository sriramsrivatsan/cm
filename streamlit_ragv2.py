import streamlit as st
import pandas as pd
import numpy as np
import openai
from io import StringIO
import json
import re
from typing import Dict, List, Any, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import nltk
import string
from datetime import datetime
import warnings
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data with error handling
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt_tab', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('words', quiet=True)
        return True
    except Exception as e:
        st.warning(f"Some NLTK data could not be downloaded: {str(e)}")
        return False

# Import NLTK components with error handling
try:
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except ImportError as e:
    st.error(f"NLTK not properly installed: {str(e)}")
    NLTK_AVAILABLE = False

# RAG-specific imports with USearch
try:
    from usearch.index import Index
    USEARCH_AVAILABLE = True
    st.sidebar.success("✅ USearch available")
except ImportError:
    USEARCH_AVAILABLE = False
    st.sidebar.error("❌ USearch not installed - install with: pip install usearch")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    st.sidebar.success("✅ SentenceTransformers available")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    st.sidebar.error("❌ SentenceTransformers not installed")

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
    st.sidebar.success("✅ tiktoken available")
except ImportError:
    TIKTOKEN_AVAILABLE = False
    st.sidebar.error("❌ tiktoken not installed")

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="CSV Natural Language Query RAG App v2.1 FIXED",
    page_icon="🧠",
    layout="wide"
)

class AdvancedTokenizer:
    def __init__(self):
        if not NLTK_AVAILABLE:
            st.error("NLTK not available - tokenization will be limited")
            return
            
        try:
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            st.error(f"Error initializing tokenizer: {str(e)}")
            self.stemmer = None
            self.lemmatizer = None
            self.stop_words = set()

    def tokenize_text(self, text: str, method='lemmatize') -> List[str]:
        """Advanced text tokenization with multiple options"""
        if pd.isna(text) or not isinstance(text, str):
            return []

        if not NLTK_AVAILABLE or not self.stemmer:
            # Fallback tokenization
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            return [token for token in text.split() if len(token) > 2]

        try:
            # Convert to lowercase and remove punctuation
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)

            # Tokenize into words
            tokens = word_tokenize(text)

            # Remove stopwords
            tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]

            # Apply stemming or lemmatization
            if method == 'stem' and self.stemmer:
                tokens = [self.stemmer.stem(token) for token in tokens]
            elif method == 'lemmatize' and self.lemmatizer:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

            # Remove empty tokens
            tokens = [token for token in tokens if token.strip()]

            return tokens
        except Exception as e:
            logger.error(f"Error in text tokenization: {str(e)}")
            # Fallback tokenization
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            return [token for token in text.split() if len(token) > 2]

    def tokenize_numeric(self, value, column_name: str) -> List[str]:
        """Tokenize numeric values with contextual information"""
        if pd.isna(value):
            return ['missing_value']

        tokens = []

        try:
            # Add the raw value as string
            tokens.append(str(value))

            # Add range-based tokens for better querying
            if isinstance(value, (int, float)):
                abs_val = abs(value)

                # Magnitude tokens
                if abs_val == 0:
                    tokens.extend(['zero', 'empty'])
                elif abs_val < 1:
                    tokens.extend(['small', 'fraction', 'decimal'])
                elif abs_val < 10:
                    tokens.extend(['single_digit', 'small'])
                elif abs_val < 100:
                    tokens.extend(['double_digit', 'medium'])
                elif abs_val < 1000:
                    tokens.extend(['hundreds', 'large'])
                elif abs_val < 10000:
                    tokens.extend(['thousands', 'very_large'])
                else:
                    tokens.extend(['huge', 'massive'])

                # Sign tokens
                if value > 0:
                    tokens.append('positive')
                elif value < 0:
                    tokens.extend(['negative', 'minus'])

                # Special number patterns
                if value == int(value):
                    tokens.append('integer')
                else:
                    tokens.extend(['decimal', 'float'])

            # Add column context
            tokens.append(f"{column_name}_value")

        except Exception as e:
            logger.error(f"Error tokenizing numeric value {value}: {str(e)}")
            tokens = [str(value), f"{column_name}_value"]

        return tokens

    def tokenize_datetime(self, value, column_name: str) -> List[str]:
        """Tokenize datetime values with temporal context"""
        if pd.isna(value):
            return ['missing_date']

        tokens = []

        try:
            if isinstance(value, str):
                dt = pd.to_datetime(value)
            else:
                dt = value

            # Basic components
            tokens.extend([
                str(dt.year), f"year_{dt.year}",
                str(dt.month), f"month_{dt.month}", dt.strftime('%B').lower(),
                str(dt.day), f"day_{dt.day}",
                dt.strftime('%A').lower(), dt.strftime('%a').lower()
            ])

            # Quarters and seasons
            quarter = (dt.month - 1) // 3 + 1
            tokens.extend([f"quarter_{quarter}", f"q{quarter}"])

            # Seasons (Northern Hemisphere)
            month = dt.month
            if month in [12, 1, 2]:
                tokens.append('winter')
            elif month in [3, 4, 5]:
                tokens.append('spring')
            elif month in [6, 7, 8]:
                tokens.append('summer')
            else:
                tokens.append('autumn')

            # Decade
            decade = (dt.year // 10) * 10
            tokens.append(f"decade_{decade}s")

            # Recent vs old
            current_year = datetime.now().year
            if dt.year == current_year:
                tokens.append('current_year')
            elif dt.year == current_year - 1:
                tokens.append('last_year')
            elif dt.year > current_year - 5:
                tokens.append('recent')
            elif dt.year < current_year - 20:
                tokens.append('old')

            # Add column context
            tokens.append(f"{column_name}_date")

        except Exception as e:
            logger.error(f"Error tokenizing datetime value {value}: {str(e)}")
            tokens = ['invalid_date', f"{column_name}_date"]

        return tokens

    def tokenize_categorical(self, value, column_name: str, value_counts: Dict = None) -> List[str]:
        """Tokenize categorical values with frequency context"""
        if pd.isna(value) or value == 'Unknown':
            return ['missing_category', 'unknown']

        tokens = []

        try:
            # Basic tokenization of the category value
            if isinstance(value, str):
                # Tokenize the category name itself
                category_tokens = self.tokenize_text(value, method='lemmatize')
                tokens.extend(category_tokens)

                # Add original value (cleaned)
                clean_value = re.sub(r'[^\w\s]', '_', str(value).lower())
                tokens.append(clean_value)

                # Add word count context
                word_count = len(value.split())
                if word_count == 1:
                    tokens.append('single_word_category')
                elif word_count > 3:
                    tokens.append('multi_word_category')

                # Add length context
                if len(value) < 5:
                    tokens.append('short_category')
                elif len(value) > 20:
                    tokens.append('long_category')

            # Add frequency context if available
            if value_counts and value in value_counts:
                total_count = sum(value_counts.values())
                frequency = value_counts[value] / total_count

                if frequency > 0.5:
                    tokens.append('dominant_category')
                elif frequency > 0.1:
                    tokens.append('common_category')
                elif frequency < 0.01:
                    tokens.append('rare_category')

            # Add column context
            tokens.append(f"{column_name}_category")

        except Exception as e:
            logger.error(f"Error tokenizing categorical value {value}: {str(e)}")
            tokens = [str(value), f"{column_name}_category"]

        return tokens

class RAGVectorStore:
    """FIXED Vector store for RAG functionality using USearch and sentence transformers"""
    
    def __init__(self):
        self.embedder = None
        self.index = None
        self.documents = []
        self.metadata = []
        self.dimension = 384  # Default for sentence transformers
        self._index_built = False
        self._initialization_error = None
        
        # Check dependencies and initialize
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self._initialization_error = "SentenceTransformers not available"
            st.error("❌ SentenceTransformers not available. Install with: pip install sentence-transformers")
            return
            
        if not USEARCH_AVAILABLE:
            self._initialization_error = "USearch not available"
            st.error("❌ USearch not available. Install with: pip install usearch")
            return
        
        # Initialize sentence transformer with error handling
        try:
            with st.spinner("Loading sentence transformer model..."):
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                # Get dimension correctly
                test_embedding = self.embedder.encode(["test"], show_progress_bar=False)
                self.dimension = len(test_embedding[0]) if len(test_embedding) > 0 else 384
                st.success("✅ Sentence transformer loaded successfully")
                logger.info(f"Loaded sentence transformer with dimension: {self.dimension}")
        except Exception as e:
            self._initialization_error = f"Sentence transformer error: {str(e)}"
            st.error(f"Error loading sentence transformer: {str(e)}")
            logger.error(f"Sentence transformer loading error: {str(e)}")
            self.embedder = None
            return
            
        # Initialize USearch index with error handling
        try:
            self._init_index()
            st.success("✅ USearch index initialized successfully")
            logger.info(f"USearch index initialized with dimension: {self.dimension}")
        except Exception as e:
            self._initialization_error = f"USearch error: {str(e)}"
            st.error(f"Error initializing USearch index: {str(e)}")
            logger.error(f"USearch index initialization error: {str(e)}")
            self.index = None
    
    def _init_index(self):
        """Initialize or reinitialize the USearch index"""
        self.index = Index(
            ndim=self.dimension,
            metric='cos',  # cosine similarity
            dtype='f32'    # float32
        )
    
    def is_available(self) -> bool:
        """Check if RAG functionality is available"""
        return (self.embedder is not None and 
                self.index is not None and 
                USEARCH_AVAILABLE and 
                SENTENCE_TRANSFORMERS_AVAILABLE and
                self._initialization_error is None)
            
    def create_document_chunks(self, processed_df: pd.DataFrame, tokenized_df: pd.DataFrame, 
                             data_summary: Dict) -> List[Dict]:
        """Create document chunks from the processed data for vector storage"""
        if not self.is_available():
            st.error("RAG system not available - cannot create document chunks")
            return []
            
        chunks = []
        
        try:
            # 1. Create column-level documents
            for col in processed_df.columns:
                col_data = processed_df[col]
                
                # Basic column info
                chunk_text = f"Column {col} contains {col_data.dtype} data with {col_data.nunique()} unique values."
                
                if col_data.dtype in ['int64', 'float64']:
                    chunk_text += f" Range: {col_data.min():.2f} to {col_data.max():.2f}, Mean: {col_data.mean():.2f}"
                elif col_data.dtype == 'object':
                    top_values = col_data.value_counts().head(3)
                    chunk_text += f" Top values: {', '.join([f'{k}({v})' for k, v in top_values.items()])}"
                
                # Add tokenization info if available
                token_col = f"{col}_tokens"
                if token_col in tokenized_df.columns:
                    sample_tokens = tokenized_df[token_col].iloc[0] if len(tokenized_df) > 0 else ""
                    if isinstance(sample_tokens, str):
                        tokens_preview = sample_tokens.split(' | ')[:5]
                        chunk_text += f" Sample tokens: {', '.join(tokens_preview)}"
                
                chunks.append({
                    'text': chunk_text,
                    'type': 'column_summary',
                    'column': col,
                    'metadata': {
                        'data_type': str(col_data.dtype),
                        'unique_count': int(col_data.nunique()),
                        'null_count': int(col_data.isnull().sum())
                    }
                })
            
            # 2. Create row-level documents (sample rows for context)
            sample_size = min(50, len(processed_df))
            sample_df = processed_df.head(sample_size)
            
            for idx, row in sample_df.iterrows():
                row_text = f"Data row {idx}: "
                row_parts = []
                
                for col in processed_df.columns[:8]:  # Limit columns for readability
                    value = row[col]
                    if pd.notna(value):
                        row_parts.append(f"{col}={value}")
                
                row_text += ", ".join(row_parts)
                
                chunks.append({
                    'text': row_text,
                    'type': 'data_row',
                    'row_index': idx,
                    'metadata': {'sample_data': True}
                })
            
            # 3. Create statistical summary documents
            if data_summary:
                summary_text = f"Dataset overview: {data_summary['dataset_info']['total_rows']} rows, "
                summary_text += f"{data_summary['dataset_info']['total_columns']} columns. "
                
                if 'tokenization_info' in data_summary and data_summary['tokenization_info']:
                    token_info = data_summary['tokenization_info']['global_stats']
                    summary_text += f"Tokenization: {token_info['total_tokens_generated']} tokens generated, "
                    summary_text += f"{token_info['total_unique_tokens']} unique tokens, "
                    summary_text += f"diversity score: {token_info['average_diversity_per_column']:.3f}"
                
                chunks.append({
                    'text': summary_text,
                    'type': 'dataset_summary',
                    'metadata': {'is_summary': True}
                })
            
            st.info(f"Created {len(chunks)} document chunks for RAG indexing")
            logger.info(f"Created {len(chunks)} document chunks")
            return chunks
            
        except Exception as e:
            st.error(f"Error creating document chunks: {str(e)}")
            logger.error(f"Document chunk creation error: {str(e)}")
            return []
    
    def build_index(self, chunks: List[Dict]) -> bool:
        """FIXED: Build USearch index from document chunks"""
        if not self.is_available():
            st.error("❌ RAG system not available. Please install required dependencies:")
            st.code("pip install usearch sentence-transformers")
            return False
            
        if not chunks:
            st.error("No document chunks provided for indexing")
            return False
            
        try:
            # Clear existing data and reinitialize index
            self.documents = []
            self.metadata = []
            self._index_built = False
            
            # Reinitialize index to ensure clean state
            self._init_index()
            
            # Process chunks in batches for memory efficiency
            batch_size = 16
            texts = [chunk['text'] for chunk in chunks]
            
            progress_bar = st.progress(0)
            st.info(f"🔄 Building vector index for RAG with {len(texts)} documents...")
            
            total_processed = 0
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_chunks = chunks[i:i + batch_size]
                
                try:
                    # Generate embeddings with proper error handling
                    embeddings = self.embedder.encode(
                        batch_texts, 
                        convert_to_tensor=False, 
                        show_progress_bar=False,
                        normalize_embeddings=True  # Normalize for better cosine similarity
                    )
                    
                    # Convert to proper format
                    if isinstance(embeddings, list):
                        embeddings = np.array(embeddings)
                    embeddings = embeddings.astype('f32')
                    
                    # Add embeddings to USearch index one by one
                    for j, embedding in enumerate(embeddings):
                        doc_id = total_processed + j
                        try:
                            # FIXED: Use correct USearch API - add one vector at a time
                            self.index.add(doc_id, embedding.flatten())
                        except Exception as add_error:
                            logger.error(f"Error adding document {doc_id} to index: {add_error}")
                            continue
                    
                    # Store documents and metadata
                    self.documents.extend(batch_texts)
                    self.metadata.extend([chunk.get('metadata', {}) for chunk in batch_chunks])
                    
                    total_processed += len(embeddings)
                    
                    # Update progress
                    progress_bar.progress(min(1.0, total_processed / len(texts)))
                    
                    # Small delay to prevent overwhelming
                    time.sleep(0.01)
                    
                except Exception as batch_error:
                    st.error(f"Error processing batch {i//batch_size + 1}: {str(batch_error)}")
                    logger.error(f"Batch processing error: {str(batch_error)}")
                    continue
            
            if total_processed > 0:
                self._index_built = True
                st.success(f"✅ Vector index built with {len(self.documents)} documents")
                logger.info(f"Vector index built successfully with {len(self.documents)} documents")
                return True
            else:
                st.error("❌ No documents were successfully indexed")
                return False
            
        except Exception as e:
            st.error(f"Error building vector index: {str(e)}")
            logger.error(f"Vector index building error: {str(e)}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """FIXED: Search for relevant documents using vector similarity"""
        if not self.is_available():
            st.warning("RAG search not available - missing dependencies")
            return []
            
        if not self._index_built or len(self.documents) == 0:
            st.warning("No documents indexed yet")
            return []
            
        try:
            # Generate query embedding
            query_embedding = self.embedder.encode(
                [query], 
                convert_to_tensor=False, 
                show_progress_bar=False,
                normalize_embeddings=True
            )
            
            # Convert to proper format
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding[0])
            else:
                query_embedding = query_embedding[0]
            query_embedding = query_embedding.astype('f32').flatten()
            
            # FIXED: Perform search with proper error handling
            try:
                search_results = self.index.search(query_embedding, k)
                logger.info(f"Search performed, result type: {type(search_results)}")
            except Exception as search_error:
                logger.error(f"Search error: {search_error}")
                return []
            
            results = []
            
            # FIXED: Handle USearch results correctly based on version
            try:
                # Try new USearch API format first
                if hasattr(search_results, 'keys') and hasattr(search_results, 'distances'):
                    doc_ids = search_results.keys
                    distances = search_results.distances
                    
                    # Convert to lists if needed
                    if hasattr(doc_ids, 'tolist'):
                        doc_ids = doc_ids.tolist()
                    if hasattr(distances, 'tolist'):
                        distances = distances.tolist()
                        
                elif isinstance(search_results, tuple) and len(search_results) == 2:
                    # Alternative format: (keys, distances)
                    doc_ids, distances = search_results
                    if hasattr(doc_ids, 'tolist'):
                        doc_ids = doc_ids.tolist()
                    if hasattr(distances, 'tolist'):
                        distances = distances.tolist()
                else:
                    # Fallback: try to access as attributes
                    doc_ids = getattr(search_results, 'keys', [])
                    distances = getattr(search_results, 'distances', [])
                
                # Process results
                for i, (doc_id, distance) in enumerate(zip(doc_ids, distances)):
                    if isinstance(doc_id, (list, np.ndarray)):
                        doc_id = doc_id[0] if len(doc_id) > 0 else 0
                    if isinstance(distance, (list, np.ndarray)):
                        distance = distance[0] if len(distance) > 0 else 1.0
                        
                    doc_id = int(doc_id)
                    distance = float(distance)
                    
                    if 0 <= doc_id < len(self.documents):
                        # FIXED: Convert cosine distance to similarity score
                        # USearch cosine distance: 0 = identical, 2 = opposite
                        similarity_score = max(0.0, 1.0 - (distance / 2.0))
                        
                        results.append({
                            'text': self.documents[doc_id],
                            'score': similarity_score,
                            'rank': i + 1,
                            'metadata': self.metadata[doc_id] if doc_id < len(self.metadata) else {},
                            'doc_id': doc_id,
                            'distance': distance
                        })
                
            except Exception as result_processing_error:
                logger.error(f"Error processing search results: {result_processing_error}")
                st.warning(f"Search completed but had issues processing results: {result_processing_error}")
                return []
            
            # Sort by score (highest first)
            results.sort(key=lambda x: x['score'], reverse=True)
            
            logger.info(f"Search completed: {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            st.error(f"Error during vector search: {str(e)}")
            logger.error(f"Vector search error: {str(e)}")
            return []
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            'available': self.is_available(),
            'index_built': self._index_built,
            'document_count': len(self.documents),
            'dimension': self.dimension,
            'metadata_count': len(self.metadata),
            'initialization_error': self._initialization_error
        }


class ConversationManager:
    """Manages conversation context and history for RAG queries"""
    
    def __init__(self):
        self.conversation_history = []
        self.max_history_length = 10  # Keep last 10 exchanges
        
    def add_exchange(self, question: str, answer: str, context_used: List[str] = None):
        """Add a question-answer exchange to the conversation history"""
        exchange = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'context_used': context_used or []
        }
        
        self.conversation_history.append(exchange)
        
        # Trim history if too long
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def get_context_for_query(self, current_question: str) -> str:
        """Get relevant conversation context for the current query"""
        if not self.conversation_history:
            return ""
        
        # Get last 3 exchanges for context
        recent_history = self.conversation_history[-3:]
        
        context_parts = ["Previous conversation context:"]
        for i, exchange in enumerate(recent_history, 1):
            context_parts.append(f"Q{i}: {exchange['question'][:100]}...")
            context_parts.append(f"A{i}: {exchange['answer'][:150]}...")
        
        return "\n".join(context_parts)
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

class OpenAIQueryProcessor:
    """FIXED: OpenAI query processor with proper error handling and token management"""
    
    def __init__(self, api_key: str):
        self._api_key = api_key
        self.client = None
        self.tokenizer = None
        self.max_context_length = 16385
        self.max_completion_tokens = 1500
        self.max_input_tokens = self.max_context_length - self.max_completion_tokens - 100
        self._token_cache = {}
        self._initialization_error = None
        
        # FIXED: Initialize OpenAI client with proper error handling
        try:
            self.client = openai.OpenAI(api_key=api_key)
            # Test the connection with a minimal call
            test_response = self.client.models.list()
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            self._initialization_error = f"OpenAI initialization error: {str(e)}"
            st.error(f"Error initializing OpenAI client: {str(e)}")
            logger.error(f"OpenAI client initialization error: {str(e)}")
            return
        
        # Initialize tokenizer for gpt-3.5-turbo with error handling
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
                logger.info("Tiktoken tokenizer initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to load tiktoken for gpt-3.5-turbo: {e}")
                try:
                    self.tokenizer = tiktoken.get_encoding("cl100k_base")  # fallback
                    logger.info("Tiktoken fallback tokenizer initialized")
                except Exception as e2:
                    logger.error(f"Failed to load fallback tiktoken encoding: {e2}")
                    self.tokenizer = None

    def is_available(self) -> bool:
        """Check if OpenAI client is properly initialized"""
        return self.client is not None and self._initialization_error is None

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string with caching"""
        if not text:
            return 0
            
        # Check cache first
        text_hash = hash(text)
        if text_hash in self._token_cache:
            return self._token_cache[text_hash]
            
        if self.tokenizer and TIKTOKEN_AVAILABLE:
            try:
                token_count = len(self.tokenizer.encode(text))
                self._token_cache[text_hash] = token_count
                return token_count
            except Exception as e:
                logger.warning(f"Error counting tokens with tiktoken: {e}")

        # Fallback: rough approximation (1 token ≈ 0.75 words)
        token_count = int(len(text.split()) * 1.3)
        self._token_cache[text_hash] = token_count
        return token_count

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        if not text or max_tokens <= 0:
            return ""
            
        current_tokens = self.count_tokens(text)
        if current_tokens <= max_tokens:
            return text

        # Split into sentences and keep as many as possible
        sentences = text.split('. ')
        truncated = ""

        for sentence in sentences:
            test_text = truncated + sentence + ". "
            if self.count_tokens(test_text) > max_tokens:
                break
            truncated = test_text

        if not truncated:  # If even first sentence is too long
            words = text.split()
            truncated = ""
            for word in words:
                test_text = truncated + " " + word if truncated else word
                if self.count_tokens(test_text) > max_tokens:
                    break
                truncated = test_text

        return truncated.strip() + ("..." if truncated else "")

    def generate_system_prompt(self, data_summary: Dict, include_tokenization: bool = True) -> str:
        """Generate an optimized system prompt based on the data summary and tokenization"""
        try:
            # Start with essential information
            dataset_info = data_summary.get('dataset_info', {})
            prompt = f"""You are an advanced data analyst assistant with access to a tokenized dataset and RAG capabilities.

Dataset: {dataset_info.get('total_rows', 0):,} rows × {dataset_info.get('total_columns', 0)} columns
Columns: {', '.join(dataset_info.get('column_names', [])[:10])}{"..." if len(dataset_info.get('column_names', [])) > 10 else ""}

"""

            # Add tokenization summary if available and requested
            if include_tokenization and data_summary.get('tokenization_info'):
                token_info = data_summary['tokenization_info']
                global_stats = token_info.get('global_stats', {})
                prompt += f"""Tokenization: {global_stats.get('total_tokens_generated', 0):,} tokens generated, {global_stats.get('total_unique_tokens', 0):,} unique, diversity: {global_stats.get('average_diversity_per_column', 0):.2f}

"""

            # Add column details (truncated)
            prompt += "Key Columns:\n"
            column_details = data_summary.get('column_details', {})
            column_count = 0
            for col, details in column_details.items():
                if column_count >= 15:  # Limit to prevent overflow
                    prompt += f"... and {len(column_details) - column_count} more columns\n"
                    break

                prompt += f"\n{col} ({details.get('data_type', 'unknown')}): {details.get('unique_count', 0)} unique"

                if 'min' in details:
                    prompt += f", range: {details['min']:.1f}-{details['max']:.1f}"
                elif 'top_values' in details:
                    top_vals = list(details['top_values'].keys())[:2]
                    prompt += f", top: {', '.join(map(str, top_vals))}"

                # Add tokenization info if available (brief)
                if 'tokenization' in details:
                    token_stats = details['tokenization']
                    prompt += f", {token_stats.get('total_tokens', 0)} tokens"

                column_count += 1

            prompt += """

RAG Capabilities: You have access to relevant document chunks retrieved via vector search. Use this context to provide more accurate and detailed responses.
Tokenization: Rich semantic tokenization enables pattern recognition and contextual insights.
Conversation: Maintain context from previous exchanges to provide coherent, building responses.
Instructions: Provide specific, actionable insights based on retrieved context and data structure."""

            # Ensure system prompt fits within reasonable limits
            max_system_tokens = int(self.max_input_tokens * 0.5)  # Reserve 50% for system prompt
            return self.truncate_text(prompt, max_system_tokens)
            
        except Exception as e:
            logger.error(f"Error generating system prompt: {str(e)}")
            # Fallback minimal prompt
            return "You are a data analyst assistant. Help analyze the provided dataset."

    def prepare_rag_context(self, retrieved_docs: List[Dict], conversation_context: str = "") -> str:
        """Prepare RAG context from retrieved documents and conversation history"""
        context_parts = []
        
        try:
            if conversation_context:
                context_parts.append(f"Conversation Context:\n{conversation_context}")
            
            if retrieved_docs:
                context_parts.append("Retrieved Information:")
                for i, doc in enumerate(retrieved_docs[:5], 1):  # Limit to top 5 results
                    score = doc.get('score', 0)
                    text = doc.get('text', '')
                    context_parts.append(f"{i}. (Score: {score:.3f}) {text}")
            
            full_context = "\n\n".join(context_parts)
            
            # Reserve space for RAG context (30% of input tokens)
            max_context_tokens = int(self.max_input_tokens * 0.3)
            return self.truncate_text(full_context, max_context_tokens)
            
        except Exception as e:
            logger.error(f"Error preparing RAG context: {str(e)}")
            return ""

    def query_data_with_rag(self, question: str, data_summary: Dict, vector_store: RAGVectorStore, 
                           conversation_manager: ConversationManager) -> str:
        """FIXED: Process natural language query with RAG and conversation context"""
        
        # Check if client is available
        if not self.is_available():
            return f"OpenAI client not available: {self._initialization_error}"
        
        try:
            # Get conversation context
            conversation_context = conversation_manager.get_context_for_query(question)
            
            # Retrieve relevant documents using vector search
            retrieved_docs = []
            try:
                retrieved_docs = vector_store.search(question, k=5)
                logger.info(f"Retrieved {len(retrieved_docs)} documents for query")
            except Exception as search_error:
                logger.warning(f"Vector search failed: {search_error}")
                # Continue without RAG context
            
            # Generate optimized system prompt
            system_prompt = self.generate_system_prompt(data_summary, include_tokenization=True)
            
            # Prepare RAG context
            rag_context = self.prepare_rag_context(retrieved_docs, conversation_context)
            
            # Create user message
            user_message = f"Question: {question}"
            if rag_context:
                user_message += f"\n\nRelevant Context:\n{rag_context}"
            
            # Final token check and adjustment
            system_tokens = self.count_tokens(system_prompt)
            user_tokens = self.count_tokens(user_message)
            
            total_input_tokens = system_tokens + user_tokens
            
            if total_input_tokens > self.max_input_tokens:
                # Reduce context further if needed
                excess_tokens = total_input_tokens - self.max_input_tokens
                if rag_context:
                    current_context_tokens = self.count_tokens(rag_context)
                    reduced_context_tokens = max(200, current_context_tokens - excess_tokens)
                    rag_context = self.truncate_text(rag_context, reduced_context_tokens)
                    user_message = f"Question: {question}\n\nRelevant Context:\n{rag_context}"
            
            # Adjust completion tokens based on remaining context
            final_input_tokens = self.count_tokens(system_prompt) + self.count_tokens(user_message)
            available_tokens = self.max_context_length - final_input_tokens - 100
            completion_tokens = min(self.max_completion_tokens, max(300, available_tokens))
            
            # FIXED: Make API call with proper error handling
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=completion_tokens,
                    temperature=0.7,
                    timeout=30  # Add timeout
                )
                
                answer = response.choices[0].message.content
                
                # Store in conversation history
                context_used = [doc['text'][:100] + "..." for doc in retrieved_docs[:3]]
                conversation_manager.add_exchange(question, answer, context_used)
                
                logger.info(f"Successfully processed RAG query: {question[:50]}...")
                return answer
                
            except openai.APIError as api_error:
                error_msg = f"OpenAI API Error: {str(api_error)}"
                logger.error(error_msg)
                return f"Error processing query: {error_msg}. Please check your API key and try again."
                
            except openai.RateLimitError as rate_error:
                error_msg = f"Rate limit exceeded: {str(rate_error)}"
                logger.error(error_msg)
                return "Rate limit exceeded. Please wait a moment and try again."
                
            except Exception as api_error:
                error_msg = f"API call failed: {str(api_error)}"
                logger.error(error_msg)
                return f"Error calling OpenAI API: {error_msg}"
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in RAG query processing: {error_msg}")
            
            if "maximum context length" in error_msg.lower():
                return """I apologize, but the dataset is too large to analyze in a single query.

Please try:
1. Ask more specific questions about particular columns
2. Use the token search feature to find specific patterns
3. Focus on summary statistics rather than detailed analysis

The dataset has been successfully processed and tokenized with RAG capabilities - you can explore it using more targeted queries."""
            else:
                return f"Error processing query: {error_msg}. Please try a simpler question or check your OpenAI API key."


class CSVProcessor:
    def __init__(self):
        self.df = None
        self.processed_df = None
        self.tokenized_df = None
        self.data_summary = None
        self.tokenization_summary = None
        self.tokenizer = AdvancedTokenizer()
        # RAG components
        self.vector_store = RAGVectorStore()
        self.conversation_manager = ConversationManager()

    def load_csv(self, uploaded_file) -> bool:
        """Load and initially process the CSV file"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    self.df = pd.read_csv(uploaded_file, encoding=encoding)
                    logger.info(f"Successfully loaded CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"Error loading CSV with {encoding}: {str(e)}")
                    continue

            if self.df is None:
                raise ValueError("Could not read file with any supported encoding")

            # Basic validation
            if self.df.empty:
                raise ValueError("CSV file is empty")
                
            if len(self.df.columns) == 0:
                raise ValueError("CSV file has no columns")

            logger.info(f"Loaded CSV: {len(self.df)} rows, {len(self.df.columns)} columns")
            return True
            
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            logger.error(f"CSV loading error: {str(e)}")
            return False

    def clean_data(self) -> bool:
        """Clean and preprocess the data"""
        if self.df is None:
            st.error("No data loaded")
            return False

        try:
            self.processed_df = self.df.copy()

            # Clean column names
            original_columns = self.processed_df.columns.tolist()
            cleaned_columns = []
            
            for col in self.processed_df.columns:
                # Clean column name
                clean_col = re.sub(r'[^\w\s]', '', str(col)).strip().replace(' ', '_').lower()
                # Ensure unique column names
                if clean_col in cleaned_columns:
                    counter = 1
                    while f"{clean_col}_{counter}" in cleaned_columns:
                        counter += 1
                    clean_col = f"{clean_col}_{counter}"
                cleaned_columns.append(clean_col)
            
            self.processed_df.columns = cleaned_columns

            # Handle missing values with better logic
            for col in self.processed_df.columns:
                if self.processed_df[col].dtype == 'object':
                    # For object columns, fill with 'Unknown'
                    self.processed_df[col] = self.processed_df[col].fillna('Unknown')
                else:
                    # For numeric columns, use median if available, otherwise 0
                    median_val = self.processed_df[col].median()
                    fill_value = median_val if pd.notna(median_val) else 0
                    self.processed_df[col] = self.processed_df[col].fillna(fill_value)

            # Convert data types more carefully
            for col in self.processed_df.columns:
                if self.processed_df[col].dtype == 'object':
                    # Try to convert to datetime first
                    try:
                        # Sample a few non-null values to test datetime conversion
                        sample_values = self.processed_df[col].dropna().head(10)
                        if len(sample_values) > 0:
                            pd.to_datetime(sample_values, errors='raise', infer_datetime_format=True)
                            self.processed_df[col] = pd.to_datetime(self.processed_df[col], errors='coerce')
                            # Fill any failed conversions
                            self.processed_df[col] = self.processed_df[col].fillna(pd.Timestamp('1900-01-01'))
                    except:
                        # Try to convert to numeric
                        try:
                            # Test conversion on sample
                            sample_values = self.processed_df[col].dropna().head(10)
                            if len(sample_values) > 0:
                                pd.to_numeric(sample_values, errors='raise')
                                self.processed_df[col] = pd.to_numeric(self.processed_df[col], errors='coerce')
                                # Fill any failed conversions
                                median_val = self.processed_df[col].median()
                                fill_value = median_val if pd.notna(median_val) else 0
                                self.processed_df[col] = self.processed_df[col].fillna(fill_value)
                        except:
                            pass  # Keep as string

            logger.info(f"Data cleaning completed: {len(self.processed_df)} rows, {len(self.processed_df.columns)} columns")
            return True
            
        except Exception as e:
            st.error(f"Error cleaning data: {str(e)}")
            logger.error(f"Data cleaning error: {str(e)}")
            return False

    def tokenize_dataset(self) -> bool:
        """Perform comprehensive tokenization of the entire dataset"""
        if self.processed_df is None:
            st.error("No processed data available")
            return False

        try:
            st.info("🔄 Starting comprehensive tokenization process...")

            # Initialize tokenized dataframe with original data
            self.tokenized_df = self.processed_df.copy()

            # Dictionary to store all tokens for summary
            all_tokens = {}
            column_token_stats = {}

            progress_bar = st.progress(0)
            total_columns = len(self.processed_df.columns)

            for idx, col in enumerate(self.processed_df.columns):
                st.info(f"Tokenizing column: {col}")

                column_tokens = []
                token_column_name = f"{col}_tokens"

                try:
                    # Determine column type and tokenize accordingly
                    if self.processed_df[col].dtype in ['int64', 'float64']:
                        # Numeric tokenization
                        for value in self.processed_df[col]:
                            tokens = self.tokenizer.tokenize_numeric(value, col)
                            column_tokens.extend(tokens)

                    elif pd.api.types.is_datetime64_any_dtype(self.processed_df[col]):
                        # Datetime tokenization
                        for value in self.processed_df[col]:
                            tokens = self.tokenizer.tokenize_datetime(value, col)
                            column_tokens.extend(tokens)

                    else:
                        # Categorical/Text tokenization
                        value_counts = self.processed_df[col].value_counts().to_dict()
                        for value in self.processed_df[col]:
                            tokens = self.tokenizer.tokenize_categorical(value, col, value_counts)
                            column_tokens.extend(tokens)

                    # Store tokens for this column
                    all_tokens[col] = column_tokens

                    # Calculate token statistics for this column
                    unique_tokens = set(column_tokens)
                    column_token_stats[col] = {
                        'total_tokens': len(column_tokens),
                        'unique_tokens': len(unique_tokens),
                        'most_common': Counter(column_tokens).most_common(10),
                        'token_diversity': len(unique_tokens) / len(column_tokens) if column_tokens else 0
                    }

                    # Add tokenized column to dataframe (store as string for display)
                    token_lists = []
                    if self.processed_df[col].dtype in ['int64', 'float64']:
                        for value in self.processed_df[col]:
                            tokens = self.tokenizer.tokenize_numeric(value, col)
                            token_lists.append(' | '.join(tokens))
                    elif pd.api.types.is_datetime64_any_dtype(self.processed_df[col]):
                        for value in self.processed_df[col]:
                            tokens = self.tokenizer.tokenize_datetime(value, col)
                            token_lists.append(' | '.join(tokens))
                    else:
                        value_counts = self.processed_df[col].value_counts().to_dict()
                        for value in self.processed_df[col]:
                            tokens = self.tokenizer.tokenize_categorical(value, col, value_counts)
                            token_lists.append(' | '.join(tokens))

                    self.tokenized_df[token_column_name] = token_lists

                except Exception as col_error:
                    logger.error(f"Error tokenizing column {col}: {str(col_error)}")
                    # Create empty tokenization for failed column
                    column_token_stats[col] = {
                        'total_tokens': 0,
                        'unique_tokens': 0,
                        'most_common': [],
                        'token_diversity': 0
                    }
                    self.tokenized_df[f"{col}_tokens"] = ['error'] * len(self.processed_df)

                # Update progress
                progress_bar.progress((idx + 1) / total_columns)

            # Create comprehensive tokenization summary
            valid_stats = {k: v for k, v in column_token_stats.items() if v['total_tokens'] > 0}
            
            self.tokenization_summary = {
                'total_columns_tokenized': len(column_token_stats),
                'successful_columns': len(valid_stats),
                'column_stats': column_token_stats,
                'global_stats': {
                    'total_tokens_generated': sum([stats['total_tokens'] for stats in valid_stats.values()]),
                    'total_unique_tokens': len(set([token for tokens in all_tokens.values() for token in tokens])),
                    'average_tokens_per_column': np.mean([stats['total_tokens'] for stats in valid_stats.values()]) if valid_stats else 0,
                    'average_diversity_per_column': np.mean([stats['token_diversity'] for stats in valid_stats.values()]) if valid_stats else 0
                }
            }

            st.success("✅ Tokenization completed successfully!")
            logger.info(f"Tokenization completed: {len(valid_stats)} successful columns")
            return True
            
        except Exception as e:
            st.error(f"Error during tokenization: {str(e)}")
            logger.error(f"Tokenization error: {str(e)}")
            return False
    
    def build_rag_index(self) -> bool:
        """Build RAG vector index from processed data"""
        if self.processed_df is None or self.tokenized_df is None:
            st.error("❌ Please process and tokenize data first")
            return False
        
        try:
            st.info("🧠 Building RAG vector index...")
            
            # Generate data summary if not already done
            if self.data_summary is None:
                self.generate_data_summary()
            
            # Create document chunks
            chunks = self.vector_store.create_document_chunks(
                self.processed_df, 
                self.tokenized_df, 
                self.data_summary
            )
            
            # Build vector index
            if self.vector_store.build_index(chunks):
                st.success("✅ RAG index built successfully!")
                logger.info("RAG index built successfully")
                return True
            else:
                st.error("❌ Failed to build RAG index")
                return False
                
        except Exception as e:
            st.error(f"Error building RAG index: {str(e)}")
            logger.error(f"RAG index building error: {str(e)}")
            return False

    def generate_data_summary(self) -> Optional[Dict]:
        """Generate a comprehensive summary of the data for the AI"""
        if self.processed_df is None:
            return None

        try:
            summary = {
                "dataset_info": {
                    "total_rows": len(self.processed_df),
                    "total_columns": len(self.processed_df.columns),
                    "column_names": list(self.processed_df.columns)
                },
                "column_details": {},
                "tokenization_info": self.tokenization_summary if self.tokenization_summary else None
            }

            for col in self.processed_df.columns:
                try:
                    col_info = {
                        "data_type": str(self.processed_df[col].dtype),
                        "null_count": int(self.processed_df[col].isnull().sum()),
                        "unique_count": int(self.processed_df[col].nunique())
                    }

                    if self.processed_df[col].dtype in ['int64', 'float64']:
                        col_info.update({
                            "min": float(self.processed_df[col].min()),
                            "max": float(self.processed_df[col].max()),
                            "mean": float(self.processed_df[col].mean()),
                            "median": float(self.processed_df[col].median())
                        })

                        # Add tokenization info if available
                        if self.tokenization_summary and col in self.tokenization_summary['column_stats']:
                            col_info['tokenization'] = self.tokenization_summary['column_stats'][col]

                    elif self.processed_df[col].dtype == 'object':
                        # Get top 5 most common values
                        top_values = self.processed_df[col].value_counts().head(5)
                        col_info["top_values"] = {str(k): int(v) for k, v in top_values.to_dict().items()}

                        # Add tokenization info if available
                        if self.tokenization_summary and col in self.tokenization_summary['column_stats']:
                            col_info['tokenization'] = self.tokenization_summary['column_stats'][col]
                            
                    summary["column_details"][col] = col_info
                    
                except Exception as col_error:
                    logger.warning(f"Error processing column {col}: {str(col_error)}")
                    # Add minimal info for failed column
                    summary["column_details"][col] = {
                        "data_type": str(self.processed_df[col].dtype),
                        "error": str(col_error)
                    }

            self.data_summary = summary
            logger.info("Data summary generated successfully")
            return summary
            
        except Exception as e:
            st.error(f"Error generating data summary: {str(e)}")
            logger.error(f"Data summary generation error: {str(e)}")
            return None


def main():
    st.title("Advanced CSV Natural Language Query RAG App v2.1 - FIXED")
    st.markdown("Upload a CSV file and ask questions about your data with **RAG (Retrieval-Augmented Generation)**, **comprehensive tokenization**, and **conversation context** for enhanced natural language understanding!")
    
    # Version indicator
    st.sidebar.markdown("### Version 2.1 - FULLY FIXED")
    st.sidebar.markdown("**Fixed Issues:**")
    st.sidebar.markdown("• Fixed OpenAI client initialization")
    st.sidebar.markdown("• Fixed USearch API usage and result handling")
    st.sidebar.markdown("• Improved error handling and debugging")
    st.sidebar.markdown("• Better token management and caching")
    st.sidebar.markdown("• Enhanced vector search with proper similarity scoring")
    st.sidebar.markdown("• Added proper timeout and rate limit handling")

    # Check dependencies with better error reporting
    missing_deps = []
    if not USEARCH_AVAILABLE:
        missing_deps.append("usearch")
        st.warning("USearch not installed. Vector search will be limited. Install with: pip install usearch")
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        missing_deps.append("sentence-transformers")
        st.warning("SentenceTransformers not installed. Install with: pip install sentence-transformers")
    if not TIKTOKEN_AVAILABLE:
        missing_deps.append("tiktoken")
        st.warning("tiktoken not installed. Token management will use approximations. Install with: pip install tiktoken")

    if missing_deps:
        st.error(f"Missing dependencies: {', '.join(missing_deps)}")
        st.code(f"pip install {' '.join(missing_deps)}")

    # Check NLTK data
    if not download_nltk_data():
        st.error("Failed to download required NLTK data. Some tokenization features may not work properly.")

    # Initialize session state
    if 'processor' not in st.session_state:
        st.session_state.processor = CSVProcessor()
    if 'openai_processor' not in st.session_state:
        st.session_state.openai_processor = None
    if 'rag_ready' not in st.session_state:
        st.session_state.rag_ready = False

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        # OpenAI API Key
        api_key = st.text_input("OpenAI API Key", type="password",
                               help="Enter your OpenAI API key")

        if api_key:
            try:
                st.session_state.openai_processor = OpenAIQueryProcessor(api_key)
                if st.session_state.openai_processor.is_available():
                    st.success("OpenAI API configured")
                else:
                    st.error(f"OpenAI configuration failed: {st.session_state.openai_processor._initialization_error}")
            except Exception as e:
                st.error(f"Error configuring OpenAI: {str(e)}")
        else:
            st.warning("Please enter your OpenAI API key")

        st.markdown("---")
        st.header("RAG Settings")
        
        rag_k_results = st.slider("Retrieved Documents", 1, 10, 5, 
                                 help="Number of relevant documents to retrieve for each query")
        
        maintain_context = st.checkbox("Maintain Conversation Context", value=True,
                                      help="Keep context from previous questions")
        
        if st.button("Clear Conversation History"):
            if hasattr(st.session_state.processor, 'conversation_manager'):
                st.session_state.processor.conversation_manager.clear_history()
                st.success("Conversation history cleared")

        st.markdown("---")
        st.header("System Status")
        
        # Show RAG system status
        if hasattr(st.session_state.processor, 'vector_store'):
            stats = st.session_state.processor.vector_store.get_stats()
            st.metric("RAG Available", "Yes" if stats['available'] else "No")
            st.metric("Index Built", "Yes" if stats['index_built'] else "No") 
            st.metric("Documents", stats['document_count'])
            st.metric("Vector Dimension", stats['dimension'])

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load the CSV
        if st.session_state.processor.load_csv(uploaded_file):
            st.success("CSV file loaded successfully!")

            # Display basic info about the original data
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", len(st.session_state.processor.df))
            with col2:
                st.metric("Columns", len(st.session_state.processor.df.columns))

            # Show original data sample
            with st.expander("View Original Data Sample"):
                st.dataframe(st.session_state.processor.df.head())

            # Data processing section
            st.header("Data Processing, Tokenization & RAG Setup")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Clean Data", type="secondary"):
                    with st.spinner("Cleaning data..."):
                        if st.session_state.processor.clean_data():
                            st.success("Data cleaned successfully!")
                        else:
                            st.error("Error cleaning data")

            with col2:
                if st.button("Tokenize Data", type="secondary"):
                    if st.session_state.processor.processed_df is not None:
                        with st.spinner("Tokenizing data..."):
                            if st.session_state.processor.tokenize_dataset():
                                st.session_state.processor.generate_data_summary()
                                st.success("Data tokenized successfully!")
                            else:
                                st.error("Error during tokenization")
                    else:
                        st.error("Please clean data first")

            with col3:
                if st.button("Build RAG Index", type="primary"):
                    if (st.session_state.processor.processed_df is not None and 
                        st.session_state.processor.tokenized_df is not None):
                        with st.spinner("Building RAG vector index..."):
                            if st.session_state.processor.build_rag_index():
                                st.session_state.rag_ready = True
                                st.success("RAG system ready!")
                            else:
                                st.error("Error building RAG index")
                    else:
                        st.error("Please clean and tokenize data first")

            # Complete workflow button
            st.markdown("### Complete Workflow")
            if st.button("Clean + Tokenize + Build RAG (All-in-One)", type="primary"):
                with st.spinner("Processing complete workflow..."):
                    success = True
                    
                    # Clean data
                    if not st.session_state.processor.clean_data():
                        st.error("Error cleaning data")
                        success = False
                    else:
                        st.success("Data cleaned!")
                    
                    # Tokenize data
                    if success and st.session_state.processor.tokenize_dataset():
                        st.session_state.processor.generate_data_summary()
                        st.success("Data tokenized!")
                    else:
                        st.error("Error during tokenization")
                        success = False
                    
                    # Build RAG index
                    if success and st.session_state.processor.build_rag_index():
                        st.session_state.rag_ready = True
                        st.success("Complete RAG system ready!")
                    else:
                        st.error("Error building RAG index")

            # Show processing status
            if st.session_state.processor.processed_df is not None:
                st.info("Data processed and cleaned")
            
            if st.session_state.processor.tokenized_df is not None:
                st.info("Data tokenized with semantic enrichment")
                
            if st.session_state.rag_ready:
                st.info("RAG vector index ready for intelligent querying")

            # Show enhanced data analysis if available
            if st.session_state.processor.tokenized_df is not None:
                st.header("Enhanced Data Analysis & Statistics")

                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Tokenization Stats", "Data Distribution", "Column Analysis"])

                with tab1:
                    st.subheader("Dataset Overview")

                    # Basic dataset statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Rows", f"{len(st.session_state.processor.processed_df):,}")
                    with col2:
                        st.metric("Total Columns", len(st.session_state.processor.processed_df.columns))
                    with col3:
                        missing_values = st.session_state.processor.processed_df.isnull().sum().sum()
                        st.metric("Missing Values", f"{missing_values:,}")
                    with col4:
                        memory_usage = st.session_state.processor.processed_df.memory_usage(deep=True).sum() / 1024**2
                        st.metric("Memory Usage", f"{memory_usage:.2f} MB")

                    # RAG Status
                    if st.session_state.rag_ready:
                        st.success("RAG System: Active with vector search capabilities")
                        st.info(f"Vector Index: {len(st.session_state.processor.vector_store.documents)} documents indexed")
                    else:
                        st.warning("RAG System: Not initialized. Build RAG index for enhanced querying.")

                with tab2:
                    st.subheader("Comprehensive Tokenization Statistics")

                    if st.session_state.processor.tokenization_summary:
                        token_summary = st.session_state.processor.tokenization_summary

                        # Global tokenization metrics
                        st.write("### Global Tokenization Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Tokens", f"{token_summary['global_stats']['total_tokens_generated']:,}")
                        with col2:
                            st.metric("Unique Tokens", f"{token_summary['global_stats']['total_unique_tokens']:,}")
                        with col3:
                            st.metric("Avg Tokens/Column", f"{token_summary['global_stats']['average_tokens_per_column']:.1f}")
                        with col4:
                            st.metric("Token Diversity", f"{token_summary['global_stats']['average_diversity_per_column']:.2f}")

                        # Column-wise tokenization breakdown
                        st.write("### Column-wise Tokenization Breakdown")

                        # Create a detailed dataframe for column statistics
                        column_stats_data = []
                        for col, stats in token_summary['column_stats'].items():
                            column_stats_data.append({
                                'Column': col,
                                'Data Type': str(st.session_state.processor.processed_df[col].dtype),
                                'Total Tokens': stats['total_tokens'],
                                'Unique Tokens': stats['unique_tokens'],
                                'Diversity Score': f"{stats['token_diversity']:.3f}",
                                'Top Token': stats['most_common'][0][0] if stats['most_common'] else 'N/A',
                                'Token Frequency': stats['most_common'][0][1] if stats['most_common'] else 0
                            })

                        stats_df = pd.DataFrame(column_stats_data)
                        st.dataframe(stats_df, use_container_width=True)

                with tab3:
                    st.subheader("Data Distribution Analysis")
                    
                    # Show basic distribution charts for numeric and categorical columns
                    numeric_cols = st.session_state.processor.processed_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        st.write("### Numeric Columns Distribution")
                        st.dataframe(st.session_state.processor.processed_df[numeric_cols].describe())

                with tab4:
                    st.subheader("Detailed Column Analysis")
                    
                    # Column selector
                    selected_column = st.selectbox("Select a column for detailed analysis:",
                                                 st.session_state.processor.processed_df.columns)

                    if selected_column:
                        col_data = st.session_state.processor.processed_df[selected_column]

                        # Basic column info
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Data Type", str(col_data.dtype))
                        with col2:
                            st.metric("Unique Values", col_data.nunique())
                        with col3:
                            st.metric("Missing Values", col_data.isnull().sum())
                        with col4:
                            st.metric("Memory Usage", f"{col_data.memory_usage(deep=True) / 1024:.2f} KB")

            # RAG-Enhanced Natural Language Query Section
            if (st.session_state.rag_ready and st.session_state.openai_processor is not None 
                and st.session_state.openai_processor.is_available()):

                st.header("RAG-Enhanced Natural Language Queries")
                st.markdown("Ask sophisticated questions with **Retrieval-Augmented Generation** and **conversation context**!")

                # RAG Status Display
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"Documents: {len(st.session_state.processor.vector_store.documents)}")
                with col2:
                    st.info(f"Context: {'Active' if maintain_context else 'Disabled'}")
                with col3:
                    history_count = len(st.session_state.processor.conversation_manager.conversation_history)
                    st.info(f"History: {history_count} exchanges")

                # Enhanced example questions
                with st.expander("Example Questions"):
                    st.markdown("""
                    **Dataset Insights with Context:**
                    - What are the main patterns in this dataset and how do they relate to business outcomes?
                    - Which columns show the strongest relationships and what insights does this provide?
                    - Based on the data structure and tokens, what analysis approach would you recommend?
                    
                    **Conversational Analysis:**
                    - Tell me about the data quality issues and suggest improvements
                    - What are the outliers in the numeric columns and what might they indicate?
                    - Compare the token diversity across different data types and explain the implications
                    """)

                # Query input with RAG capabilities
                question = st.text_area("Ask a question about your data (RAG-enhanced with context):",
                                       placeholder="e.g., What are the key insights from this dataset and how should I approach the analysis?",
                                       height=100)

                # Query options
                col1, col2 = st.columns(2)
                with col1:
                    show_retrieved_docs = st.checkbox("Show Retrieved Context", value=True,
                                                    help="Display the documents retrieved by the RAG system")
                with col2:
                    show_conversation_context = st.checkbox("Show Conversation Context", value=False,
                                                          help="Display the conversation history context")

                if st.button("Query with RAG Enhancement") and question:
                    with st.spinner("Processing RAG-enhanced query with conversation context..."):
                        # Perform RAG-enhanced query
                        response = st.session_state.openai_processor.query_data_with_rag(
                            question,
                            st.session_state.processor.data_summary,
                            st.session_state.processor.vector_store,
                            st.session_state.processor.conversation_manager
                        )

                        # Display results
                        st.subheader("RAG-Enhanced Analysis Results")
                        st.write(response)

                        # Show retrieved context if requested
                        if show_retrieved_docs:
                            with st.expander("Retrieved Context Documents"):
                                retrieved_docs = st.session_state.processor.vector_store.search(question, k=rag_k_results)
                                for i, doc in enumerate(retrieved_docs, 1):
                                    st.write(f"**Document {i}** (Similarity: {doc['score']:.3f})")
                                    st.write(doc['text'])
                                    if doc.get('metadata'):
                                        st.caption(f"Metadata: {doc['metadata']}")
                                    st.markdown("---")

                        # Show conversation context if requested
                        if show_conversation_context and maintain_context:
                            with st.expander("Conversation Context"):
                                history = st.session_state.processor.conversation_manager.conversation_history
                                for i, exchange in enumerate(reversed(history[-3:]), 1):
                                    st.write(f"**Exchange {len(history) - i + 1}:**")
                                    st.write(f"*Q:* {exchange['question'][:150]}...")
                                    st.write(f"*A:* {exchange['answer'][:200]}...")
                                    st.markdown("---")

                # RAG Testing and Exploration
                st.header("RAG System Testing & Exploration")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Vector Search Test")
                    test_query = st.text_input("Test semantic search:", 
                                             placeholder="e.g., numeric columns with high variance")
                    
                    if test_query and st.button("Search Documents"):
                        results = st.session_state.processor.vector_store.search(test_query, k=5)
                        if results:
                            for i, result in enumerate(results, 1):
                                st.write(f"**{i}.** Score: {result['score']:.3f}")
                                st.write(result['text'][:200] + "...")
                                st.markdown("---")
                        else:
                            st.write("No results found")

                with col2:
                    st.subheader("Conversation History")
                    history = st.session_state.processor.conversation_manager.conversation_history
                    if history:
                        for i, exchange in enumerate(reversed(history[-5:]), 1):
                            with st.expander(f"Exchange {len(history) - i + 1}: {exchange['question'][:50]}..."):
                                st.write(f"**Question:** {exchange['question']}")
                                st.write(f"**Answer:** {exchange['answer'][:300]}...")
                                st.caption(f"Time: {exchange['timestamp']}")
                    else:
                        st.write("No conversation history yet")

            elif st.session_state.processor.tokenized_df is not None and st.session_state.openai_processor is not None:
                st.header("RAG System Not Ready")
                if not st.session_state.openai_processor.is_available():
                    st.error(f"OpenAI client not available: {st.session_state.openai_processor._initialization_error}")
                else:
                    st.warning("Please build the RAG index to enable enhanced querying with retrieval capabilities.")

            # Export functionality
            if st.session_state.processor.tokenized_df is not None:
                st.header("Export Enhanced Data & RAG Components")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("Download Processed Data"):
                        csv = st.session_state.processor.processed_df.to_csv(index=False)
                        st.download_button(
                            label="Download Processed CSV",
                            data=csv,
                            file_name="processed_data.csv",
                            mime="text/csv"
                        )

                with col2:
                    if st.button("Download Tokenized Data"):
                        csv = st.session_state.processor.tokenized_df.to_csv(index=False)
                        st.download_button(
                            label="Download Tokenized CSV",
                            data=csv,
                            file_name="tokenized_data.csv",
                            mime="text/csv"
                        )

                with col3:
                    if st.button("Download RAG Documents"):
                        if st.session_state.processor.vector_store.documents:
                            rag_data = {
                                "documents": st.session_state.processor.vector_store.documents,
                                "metadata": st.session_state.processor.vector_store.metadata,
                                "total_documents": len(st.session_state.processor.vector_store.documents)
                            }
                            json_data = json.dumps(rag_data, indent=2)
                            st.download_button(
                                label="Download RAG Documents JSON",
                                data=json_data,
                                file_name="rag_documents.json",
                                mime="application/json"
                            )

    # Footer with version 2.1 features
    st.markdown("---")
    st.markdown("""
    ## Version 2.1 - FULLY FIXED - RAG-Enhanced Features

    **Fixed Issues in This Version:**
    - **OpenAI Client**: Proper initialization with error handling and connection testing
    - **USearch API**: Fixed search result handling with multiple format support
    - **Error Handling**: Comprehensive try-catch blocks and graceful fallbacks
    - **Token Management**: Improved counting with caching and better approximations
    - **Vector Search**: Fixed similarity scoring and result processing
    - **Memory Management**: Better resource cleanup and batch processing
    - **Rate Limiting**: Added timeout and proper API error handling

    **Dependencies for Full Functionality:**
    ```bash
    pip install streamlit pandas numpy openai plotly nltk tiktoken usearch sentence-transformers
    ```
    
    **How to Run:**
    ```bash
    # Save all parts as csv_rag_app_v2_1_fixed.py, then run:
    streamlit run csv_rag_app_v2_1_fixed.py
    ```
    """)

if __name__ == "__main__":
    main()

