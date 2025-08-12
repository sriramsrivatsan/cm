import streamlit as st
import pandas as pd
import numpy as np
import openai
from io import StringIO
import json
import re
from typing import Dict, List, Any, Tuple
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import nltk
import string
from datetime import datetime
import warnings

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
        st.warning(f"⚠️ Some NLTK data could not be downloaded: {str(e)}")
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
    st.error(f"NLTK import error: {str(e)}")
    NLTK_AVAILABLE = False

# RAG-specific imports
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="CSV Natural Language Query RAG App v2.0",
    page_icon="🧠",
    layout="wide"
)

#Part 2

class AdvancedTokenizer:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = None
        self.stop_words = set()

        # Initialize components with error handling
        try:
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            st.warning(f"⚠️ WordNet lemmatizer not available: {str(e)}. Using stemming fallback.")
            self.lemmatizer = None

        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            st.warning(f"⚠️ Stopwords not available: {str(e)}. Using basic stopwords.")
            # Basic fallback stopwords
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])

    def tokenize_text(self, text: str, method='lemmatize') -> List[str]:
        """Advanced text tokenization with multiple options and error handling"""
        if pd.isna(text) or not isinstance(text, str):
            return []

        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)

        # Tokenize into words - use fallback if NLTK fails
        try:
            tokens = word_tokenize(text)
        except Exception:
            # Fallback: simple split
            tokens = text.split()

        # Remove stopwords and short tokens
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]

        # Apply stemming or lemmatization with fallbacks
        processed_tokens = []
        for token in tokens:
            try:
                if method == 'stem':
                    processed_tokens.append(self.stemmer.stem(token))
                elif method == 'lemmatize' and self.lemmatizer is not None:
                    processed_tokens.append(self.lemmatizer.lemmatize(token))
                else:
                    # Fallback: use stemming or original token
                    try:
                        processed_tokens.append(self.stemmer.stem(token))
                    except:
                        processed_tokens.append(token)
            except Exception:
                # If all else fails, use the original token
                processed_tokens.append(token)

        # Remove empty tokens
        processed_tokens = [token for token in processed_tokens if token.strip()]

        return processed_tokens

    def tokenize_numeric(self, value, column_name: str) -> List[str]:
        """Tokenize numeric values with contextual information"""
        if pd.isna(value):
            return ['missing_value']

        tokens = []

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

        except:
            tokens = ['invalid_date']

        return tokens

    def tokenize_categorical(self, value, column_name: str, value_counts: Dict = None) -> List[str]:
        """Tokenize categorical values with frequency context and error handling"""
        if pd.isna(value) or value == 'Unknown':
            return ['missing_category', 'unknown']

        tokens = []

        # Basic tokenization of the category value
        if isinstance(value, str):
            try:
                # Tokenize the category name itself with error handling
                category_tokens = self.tokenize_text(value, method='lemmatize')
                tokens.extend(category_tokens)
            except Exception:
                # Fallback: simple processing
                clean_tokens = re.sub(r'[^\w\s]', ' ', str(value).lower()).split()
                tokens.extend([token for token in clean_tokens if len(token) > 2])

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

        return tokens


class RAGVectorStore:
    """Vector store for RAG functionality using FAISS and sentence transformers"""

    def __init__(self):
        self.embedder = None
        self.index = None
        self.documents = []
        self.metadata = []
        self.dimension = 384  # Default for sentence transformers

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                self.dimension = 384
            except Exception as e:
                st.error(f"Error loading sentence transformer: {str(e)}")

        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for similarity

    def create_document_chunks(self, processed_df: pd.DataFrame, tokenized_df: pd.DataFrame,
                             data_summary: Dict) -> List[Dict]:
        """Create document chunks from the processed data for vector storage"""
        chunks = []

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
        sample_size = min(100, len(processed_df))  # Limit to 100 rows for performance
        sample_df = processed_df.head(sample_size)

        for idx, row in sample_df.iterrows():
            row_text = f"Data row {idx}: "
            row_parts = []

            for col in processed_df.columns[:10]:  # Limit columns for readability
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

        return chunks

    def build_index(self, chunks: List[Dict]):
        """Build FAISS index from document chunks"""
        if not self.embedder or not FAISS_AVAILABLE:
            st.warning("⚠️ Vector search not available. Missing dependencies.")
            return False

        try:
            # Clear existing data
            self.documents = []
            self.metadata = []
            self.index = faiss.IndexFlatIP(self.dimension)

            # Process chunks in batches for memory efficiency
            batch_size = 32
            texts = [chunk['text'] for chunk in chunks]

            progress_bar = st.progress(0)
            st.info("🔄 Building vector index for RAG...")

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_chunks = chunks[i:i + batch_size]

                # Generate embeddings
                embeddings = self.embedder.encode(batch_texts, convert_to_tensor=False)
                embeddings = np.array(embeddings).astype('float32')

                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embeddings)

                # Add to index
                self.index.add(embeddings)

                # Store documents and metadata
                self.documents.extend(batch_texts)
                self.metadata.extend([chunk.get('metadata', {}) for chunk in batch_chunks])

                # Update progress
                progress_bar.progress(min(1.0, (i + batch_size) / len(texts)))

            st.success(f"✅ Vector index built with {len(self.documents)} documents")
            return True

        except Exception as e:
            st.error(f"Error building vector index: {str(e)}")
            return False

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant documents using vector similarity"""
        if not self.embedder or not self.index or len(self.documents) == 0:
            return []

        try:
            # Generate query embedding
            query_embedding = self.embedder.encode([query], convert_to_tensor=False)
            query_embedding = np.array(query_embedding).astype('float32')
            faiss.normalize_L2(query_embedding)

            # Search
            scores, indices = self.index.search(query_embedding, k)

            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.documents):  # Valid index
                    results.append({
                        'text': self.documents[idx],
                        'score': float(score),
                        'rank': i + 1,
                        'metadata': self.metadata[idx] if idx < len(self.metadata) else {}
                    })

            return results

        except Exception as e:
            st.error(f"Error during vector search: {str(e)}")
            return []

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

    def load_csv(self, uploaded_file):
        """Load and initially process the CSV file"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    self.df = pd.read_csv(uploaded_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if self.df is None:
                raise ValueError("Could not read file with any supported encoding")

            return True
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            return False

    def clean_data(self):
        """Clean and preprocess the data"""
        if self.df is None:
            return False

        self.processed_df = self.df.copy()

        # Clean column names
        original_columns = self.processed_df.columns.tolist()
        self.processed_df.columns = [
            re.sub(r'[^\w\s]', '', col).strip().replace(' ', '_').lower()
            for col in self.processed_df.columns
        ]

        # Handle missing values
        for col in self.processed_df.columns:
            if self.processed_df[col].dtype == 'object':
                self.processed_df[col] = self.processed_df[col].fillna('Unknown')
            else:
                self.processed_df[col] = self.processed_df[col].fillna(self.processed_df[col].median())

        # Convert data types appropriately
        for col in self.processed_df.columns:
            # Try to convert to numeric if possible
            if self.processed_df[col].dtype == 'object':
                # Check if it's a date
                try:
                    pd.to_datetime(self.processed_df[col], errors='raise', infer_datetime_format=True)
                    self.processed_df[col] = pd.to_datetime(self.processed_df[col])
                except:
                    # Try to convert to numeric
                    try:
                        self.processed_df[col] = pd.to_numeric(self.processed_df[col], errors='raise')
                    except:
                        pass  # Keep as string

        return True

    def tokenize_dataset(self):
        """Perform comprehensive tokenization of the entire dataset"""
        if self.processed_df is None:
            return False

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

            # Update progress
            progress_bar.progress((idx + 1) / total_columns)

        # Create comprehensive tokenization summary
        self.tokenization_summary = {
            'total_columns_tokenized': len(column_token_stats),
            'column_stats': column_token_stats,
            'global_stats': {
                'total_tokens_generated': sum([stats['total_tokens'] for stats in column_token_stats.values()]),
                'total_unique_tokens': len(set([token for tokens in all_tokens.values() for token in tokens])),
                'average_tokens_per_column': np.mean([stats['total_tokens'] for stats in column_token_stats.values()]),
                'average_diversity_per_column': np.mean([stats['token_diversity'] for stats in column_token_stats.values()])
            }
        }

        st.success("✅ Tokenization completed successfully!")
        return True

    def build_rag_index(self):
        """Build RAG vector index from processed data"""
        if self.processed_df is None or self.tokenized_df is None:
            st.error("❌ Please process and tokenize data first")
            return False

        st.info("🧠 Building RAG vector index...")

        # Create document chunks
        chunks = self.vector_store.create_document_chunks(
            self.processed_df,
            self.tokenized_df,
            self.data_summary
        )

        # Build vector index
        if self.vector_store.build_index(chunks):
            st.success("✅ RAG index built successfully!")
            return True
        else:
            st.error("❌ Failed to build RAG index")
            return False

    def generate_data_summary(self):
        """Generate a comprehensive summary of the data for the AI"""
        if self.processed_df is None:
            return None

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
            col_info = {
                "data_type": str(self.processed_df[col].dtype),
                "null_count": int(self.processed_df[col].isnull().sum()),
                "unique_count": int(self.processed_df[col].nunique())
            }

            try:
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
            except Exception as e:
                st.warning(f"Error processing column {col}: {str(e)}")
                continue

            summary["column_details"][col] = col_info

        self.data_summary = summary
        return summary

class OpenAIQueryProcessor:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)
        # Initialize tokenizer for gpt-3.5-turbo
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            except:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")  # fallback
        else:
            self.tokenizer = None

        self.max_context_length = 16385
        self.max_completion_tokens = 1500
        self.max_input_tokens = self.max_context_length - self.max_completion_tokens - 100  # safety buffer

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string"""
        if self.tokenizer and TIKTOKEN_AVAILABLE:
            try:
                return len(self.tokenizer.encode(text))
            except:
                pass

        # Fallback: rough approximation (1 token ≈ 0.75 words)
        return int(len(text.split()) * 1.3)

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        if self.count_tokens(text) <= max_tokens:
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
                test_text = truncated + " " + word
                if self.count_tokens(test_text) > max_tokens:
                    break
                truncated = test_text

        return truncated.strip() + "..."

    def generate_system_prompt(self, data_summary: Dict, include_tokenization: bool = True) -> str:
        """Generate an optimized system prompt based on the data summary and tokenization"""
        # Start with essential information
        prompt = f"""You are an advanced data analyst assistant with access to a tokenized dataset and RAG capabilities.

Dataset: {data_summary['dataset_info']['total_rows']:,} rows × {data_summary['dataset_info']['total_columns']} columns
Columns: {', '.join(data_summary['dataset_info']['column_names'][:10])}{"..." if len(data_summary['dataset_info']['column_names']) > 10 else ""}

"""

        # Add tokenization summary if available and requested
        if include_tokenization and data_summary.get('tokenization_info'):
            token_info = data_summary['tokenization_info']
            prompt += f"""Tokenization: {token_info['global_stats']['total_tokens_generated']:,} tokens generated, {token_info['global_stats']['total_unique_tokens']:,} unique, diversity: {token_info['global_stats']['average_diversity_per_column']:.2f}

"""

        # Add column details (truncated)
        prompt += "Key Columns:\n"
        column_count = 0
        for col, details in data_summary['column_details'].items():
            if column_count >= 15:  # Limit to prevent overflow
                prompt += f"... and {len(data_summary['column_details']) - column_count} more columns\n"
                break

            prompt += f"\n{col} ({details['data_type']}): {details['unique_count']} unique"

            if 'min' in details:
                prompt += f", range: {details['min']:.1f}-{details['max']:.1f}"
            elif 'top_values' in details:
                top_vals = list(details['top_values'].keys())[:2]
                prompt += f", top: {', '.join(map(str, top_vals))}"

            # Add tokenization info if available (brief)
            if 'tokenization' in details:
                token_stats = details['tokenization']
                prompt += f", {token_stats['total_tokens']} tokens"

            column_count += 1

        prompt += """

RAG Capabilities: You have access to relevant document chunks retrieved via vector search. Use this context to provide more accurate and detailed responses.
Tokenization: Rich semantic tokenization enables pattern recognition and contextual insights.
Conversation: Maintain context from previous exchanges to provide coherent, building responses.
Instructions: Provide specific, actionable insights based on retrieved context and data structure."""

        # Ensure system prompt fits within reasonable limits
        max_system_tokens = int(self.max_input_tokens * 0.5)  # Reserve 50% for system prompt
        return self.truncate_text(prompt, max_system_tokens)

    def prepare_rag_context(self, retrieved_docs: List[Dict], conversation_context: str = "") -> str:
        """Prepare RAG context from retrieved documents and conversation history"""
        context_parts = []

        if conversation_context:
            context_parts.append(f"Conversation Context:\n{conversation_context}")

        if retrieved_docs:
            context_parts.append("Retrieved Information:")
            for i, doc in enumerate(retrieved_docs[:5], 1):  # Limit to top 5 results
                context_parts.append(f"{i}. (Score: {doc['score']:.3f}) {doc['text']}")

        full_context = "\n\n".join(context_parts)

        # Reserve space for RAG context (50% of input tokens)
        max_context_tokens = int(self.max_input_tokens * 0.5)
        return self.truncate_text(full_context, max_context_tokens)

    def query_data_with_rag(self, question: str, data_summary: Dict, vector_store: RAGVectorStore,
                           conversation_manager: ConversationManager) -> str:
        """Process natural language query with RAG and conversation context"""

        # Get conversation context
        conversation_context = conversation_manager.get_context_for_query(question)

        # Retrieve relevant documents using vector search
        retrieved_docs = vector_store.search(question, k=5)

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

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=completion_tokens,
                temperature=0.7
            )

            answer = response.choices[0].message.content

            # Store in conversation history
            context_used = [doc['text'][:100] + "..." for doc in retrieved_docs[:3]]
            conversation_manager.add_exchange(question, answer, context_used)

            return answer

        except Exception as e:
            error_msg = str(e)
            if "maximum context length" in error_msg.lower():
                return """I apologize, but the dataset is too large to analyze in a single query.

Please try:
1. Ask more specific questions about particular columns
2. Use the token search feature to find specific patterns
3. Focus on summary statistics rather than detailed analysis

The dataset has been successfully processed and tokenized with RAG capabilities - you can explore it using more targeted queries."""
            else:
                return f"Error processing query: {error_msg}"

def main():
    st.title("🧠 Advanced CSV Natural Language Query RAG App v2.0")
    st.markdown("Upload a CSV file and ask questions about your data with **RAG (Retrieval-Augmented Generation)**, **comprehensive tokenization**, and **conversation context** for enhanced natural language understanding!")

    # Version indicator
    st.sidebar.markdown("### 🏷️ Version 2.0")
    st.sidebar.markdown("**New Features:**")
    st.sidebar.markdown("• RAG with vector search")
    st.sidebar.markdown("• Conversation context")
    st.sidebar.markdown("• Enhanced query processing")
    st.sidebar.markdown("• Semantic document retrieval")

    # Check dependencies
    missing_deps = []
    if not FAISS_AVAILABLE:
        missing_deps.append("faiss-cpu")
        st.warning("⚠️ FAISS not installed. Vector search will be limited. Install with: pip install faiss-cpu")
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        missing_deps.append("sentence-transformers")
        st.warning("⚠️ SentenceTransformers not installed. Install with: pip install sentence-transformers")
    if not TIKTOKEN_AVAILABLE:
        missing_deps.append("tiktoken")
        st.warning("⚠️ tiktoken not installed. Token management will use approximations. Install with: pip install tiktoken")

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
            st.session_state.openai_processor = OpenAIQueryProcessor(api_key)
            st.success("✅ OpenAI API configured")
        else:
            st.warning("⚠️ Please enter your OpenAI API key")

        st.markdown("---")
        st.header("RAG Settings")

        rag_k_results = st.slider("Retrieved Documents", 1, 10, 5,
                                 help="Number of relevant documents to retrieve for each query")

        maintain_context = st.checkbox("Maintain Conversation Context", value=True,
                                      help="Keep context from previous questions")

        if st.button("🔄 Clear Conversation History"):
            if hasattr(st.session_state.processor, 'conversation_manager'):
                st.session_state.processor.conversation_manager.clear_history()
                st.success("✅ Conversation history cleared")

        st.markdown("---")
        st.header("Tokenization Settings")

        tokenization_method = st.selectbox(
            "Text Processing Method",
            ["lemmatize", "stem", "basic"],
            help="Choose how to process text tokens"
        )

        show_token_details = st.checkbox("Show detailed token analysis", value=True)

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load the CSV
        if st.session_state.processor.load_csv(uploaded_file):
            st.success("✅ CSV file loaded successfully!")

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
            st.header("🔧 Data Processing, Tokenization & RAG Setup")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Clean Data", type="secondary"):
                    with st.spinner("Cleaning data..."):
                        if st.session_state.processor.clean_data():
                            st.success("✅ Data cleaned successfully!")
                        else:
                            st.error("❌ Error cleaning data")

            with col2:
                if st.button("🏷️ Tokenize Data", type="secondary"):
                    if st.session_state.processor.processed_df is not None:
                        with st.spinner("Tokenizing data..."):
                            if st.session_state.processor.tokenize_dataset():
                                st.session_state.processor.generate_data_summary()
                                st.success("✅ Data tokenized successfully!")
                            else:
                                st.error("❌ Error during tokenization")
                    else:
                        st.error("❌ Please clean data first")

            with col3:
                if st.button("🧠 Build RAG Index", type="primary"):
                    if (st.session_state.processor.processed_df is not None and
                        st.session_state.processor.tokenized_df is not None):
                        with st.spinner("Building RAG vector index..."):
                            if st.session_state.processor.build_rag_index():
                                st.session_state.rag_ready = True
                                st.balloons()
                                st.success("🎉 RAG system ready!")
                            else:
                                st.error("❌ Error building RAG index")
                    else:
                        st.error("❌ Please clean and tokenize data first")

            # Complete workflow button
            st.markdown("### 🚀 Complete Workflow")
            if st.button("⚡ Clean + Tokenize + Build RAG (All-in-One)", type="primary"):
                with st.spinner("Processing complete workflow..."):
                    success = True

                    # Clean data
                    if not st.session_state.processor.clean_data():
                        st.error("❌ Error cleaning data")
                        success = False
                    else:
                        st.success("✅ Data cleaned!")

                    # Tokenize data
                    if success and st.session_state.processor.tokenize_dataset():
                        st.session_state.processor.generate_data_summary()
                        st.success("✅ Data tokenized!")
                    else:
                        st.error("❌ Error during tokenization")
                        success = False

                    # Build RAG index
                    if success and st.session_state.processor.build_rag_index():
                        st.session_state.rag_ready = True
                        st.balloons()
                        st.success("🎉 Complete RAG system ready!")
                    else:
                        st.error("❌ Error building RAG index")

            # Show processing status
            if st.session_state.processor.processed_df is not None:
                st.info("📊 Data processed and cleaned")

            if st.session_state.processor.tokenized_df is not None:
                st.info("🏷️ Data tokenized with semantic enrichment")

            if st.session_state.rag_ready:
                st.info("🧠 RAG vector index ready for intelligent querying")

# Show processed and tokenized data info
            if st.session_state.processor.tokenized_df is not None:
                st.header("📈 Enhanced Data Analysis & Statistics")

                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🏷️ Tokenization Stats", "📈 Data Distribution", "🔍 Column Analysis"])

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
                        st.success("🧠 RAG System: Active with vector search capabilities")
                        st.info(f"📚 Vector Index: {len(st.session_state.processor.vector_store.documents)} documents indexed")
                    else:
                        st.warning("⚠️ RAG System: Not initialized. Build RAG index for enhanced querying.")

                    # Data type distribution
                    st.subheader("Data Type Distribution")
                    dtype_counts = st.session_state.processor.processed_df.dtypes.value_counts()

                    col1, col2 = st.columns(2)
                    with col1:
                        # Convert dtype names to strings to avoid JSON serialization issues
                        dtype_names = [str(dtype) for dtype in dtype_counts.index]
                        fig_dtype = px.pie(values=dtype_counts.values, names=dtype_names,
                                          title="Column Data Types")
                        st.plotly_chart(fig_dtype, use_container_width=True)

                    with col2:
                        st.write("**Data Type Breakdown:**")
                        for dtype, count in dtype_counts.items():
                            percentage = (count / len(st.session_state.processor.processed_df.columns)) * 100
                            st.write(f"• {str(dtype)}: {count} columns ({percentage:.1f}%)")

                with tab2:
                    st.subheader("Comprehensive Tokenization Statistics")

                    if st.session_state.processor.tokenization_summary:
                        token_summary = st.session_state.processor.tokenization_summary

                        # Global tokenization metrics
                        st.write("### 🌐 Global Tokenization Metrics")
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
                        st.write("### 📋 Column-wise Tokenization Breakdown")

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

                        # Tokenization distribution charts
                        st.write("### 📊 Tokenization Distribution Analysis")

                        col1, col2 = st.columns(2)

                        with col1:
                            # Token count distribution
                            columns = list(token_summary['column_stats'].keys())
                            token_counts = [token_summary['column_stats'][col]['total_tokens'] for col in columns]

                            fig_tokens = px.bar(x=columns, y=token_counts,
                                              title="Token Count by Column",
                                              labels={'x': 'Columns', 'y': 'Token Count'},
                                              color=token_counts,
                                              color_continuous_scale='blues')
                            fig_tokens.update_layout(xaxis_tickangle=45)
                            st.plotly_chart(fig_tokens, use_container_width=True)

                        with col2:
                            # Diversity score distribution
                            diversity_scores = [token_summary['column_stats'][col]['token_diversity'] for col in columns]

                            fig_diversity = px.bar(x=columns, y=diversity_scores,
                                                 title="Token Diversity by Column",
                                                 labels={'x': 'Columns', 'y': 'Diversity Score'},
                                                 color=diversity_scores,
                                                 color_continuous_scale='viridis')
                            fig_diversity.update_layout(xaxis_tickangle=45)
                            st.plotly_chart(fig_diversity, use_container_width=True)

                with tab3:
                    st.subheader("Data Distribution Analysis")

                    # Numeric columns analysis
                    numeric_cols = st.session_state.processor.processed_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        st.write("### 📊 Numeric Columns Distribution")

                        # Statistical summary
                        st.write("**Statistical Summary:**")
                        st.dataframe(st.session_state.processor.processed_df[numeric_cols].describe())

                        # Distribution plots for top numeric columns
                        for i, col in enumerate(numeric_cols[:2]):  # Show first 2 numeric columns
                            col1, col2 = st.columns(2)

                            with col1:
                                # Convert to list to avoid dtype serialization issues
                                data_values = st.session_state.processor.processed_df[col].dropna().tolist()
                                fig_hist = px.histogram(x=data_values, title=f"Distribution of {col}")
                                st.plotly_chart(fig_hist, use_container_width=True)

                            with col2:
                                fig_box = px.box(y=data_values, title=f"Box Plot of {col}")
                                st.plotly_chart(fig_box, use_container_width=True)

                    # Categorical columns analysis
                    cat_cols = st.session_state.processor.processed_df.select_dtypes(include=['object']).columns
                    if len(cat_cols) > 0:
                        st.write("### 🏷️ Categorical Columns Distribution")

                        for col in cat_cols[:2]:  # Show first 2 categorical columns
                            if st.session_state.processor.processed_df[col].nunique() <= 20:
                                value_counts = st.session_state.processor.processed_df[col].value_counts().head(10)

                                col1, col2 = st.columns(2)

                                with col1:
                                    # Convert to lists for plotly compatibility
                                    value_list = value_counts.index.tolist()
                                    count_list = value_counts.values.tolist()
                                    fig_bar = px.bar(x=value_list, y=count_list,
                                                   title=f"Top Values in {col}")
                                    fig_bar.update_layout(xaxis_tickangle=45)
                                    st.plotly_chart(fig_bar, use_container_width=True)

                                with col2:
                                    fig_pie = px.pie(values=count_list, names=value_list,
                                                   title=f"Distribution of {col}")
                                    st.plotly_chart(fig_pie, use_container_width=True)
                            else:
                                st.write(f"**{col}**: {st.session_state.processor.processed_df[col].nunique()} unique values (too many to display)")

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

                        # Tokenization info for selected column
                        if (st.session_state.processor.tokenization_summary and
                            selected_column in st.session_state.processor.tokenization_summary['column_stats']):

                            token_stats = st.session_state.processor.tokenization_summary['column_stats'][selected_column]

                            st.write("### 🏷️ Tokenization Details")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Tokens", token_stats['total_tokens'])
                            with col2:
                                st.metric("Unique Tokens", token_stats['unique_tokens'])
                            with col3:
                                st.metric("Diversity Score", f"{token_stats['token_diversity']:.3f}")

                            # Most common tokens
                            st.write("**Most Common Tokens:**")
                            token_df = pd.DataFrame(token_stats['most_common'][:15], columns=['Token', 'Frequency'])

                            col1, col2 = st.columns(2)
                            with col1:
                                st.dataframe(token_df)
                            with col2:
                                if len(token_df) > 0:
                                    fig_tokens = px.bar(token_df, x='Token', y='Frequency',
                                                      title=f"Top Tokens in {selected_column}")
                                    fig_tokens.update_layout(xaxis_tickangle=45)
                                    st.plotly_chart(fig_tokens, use_container_width=True)

                # Show tokenized data sample
                with st.expander("🔍 View Tokenized Data Sample"):
                    # Show original vs tokenized side by side
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original Data")
                        st.dataframe(st.session_state.processor.processed_df.head(3))
                    with col2:
                        st.subheader("Tokenized Representation")
                        # Show only token columns
                        token_columns = [col for col in st.session_state.processor.tokenized_df.columns if col.endswith('_tokens')]
                        if token_columns:
                            st.dataframe(st.session_state.processor.tokenized_df[token_columns].head(3))

# RAG-Enhanced Natural Language Query Section
            if (st.session_state.rag_ready and st.session_state.openai_processor is not None):

                st.header("🧠 RAG-Enhanced Natural Language Queries")
                st.markdown("Ask sophisticated questions with **Retrieval-Augmented Generation** and **conversation context**!")

                # RAG Status Display
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"📚 Documents: {len(st.session_state.processor.vector_store.documents)}")
                with col2:
                    st.info(f"💬 Context: {'Active' if maintain_context else 'Disabled'}")
                with col3:
                    history_count = len(st.session_state.processor.conversation_manager.conversation_history)
                    st.info(f"📝 History: {history_count} exchanges")

                # Enhanced example questions
                with st.expander("💡 RAG-Enhanced Example Questions"):
                    st.markdown("""
                    **Dataset Insights with Context:**
                    - What are the main patterns in this dataset and how do they relate to business outcomes?
                    - Which columns show the strongest relationships and what insights does this provide?
                    - Based on the data structure and tokens, what analysis approach would you recommend?

                    **Conversational Analysis:**
                    - Tell me about the data quality issues and suggest improvements
                    - Follow-up: How would these improvements affect my analysis strategy?
                    - What are the outliers in the numeric columns and what might they indicate?
                    - Follow-up: Should I remove these outliers or investigate them further?

                    **Advanced RAG Queries:**
                    - Compare the token diversity across different data types and explain the implications
                    - What semantic patterns emerge from the categorical tokenization that might indicate data relationships?
                    - Based on statistical analysis and token patterns, identify potential data quality concerns
                    - Suggest a comprehensive analysis workflow for this specific dataset structure
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

                if st.button("🧠 Query with RAG Enhancement") and question:
                    with st.spinner("Processing RAG-enhanced query with conversation context..."):
                        # Perform RAG-enhanced query
                        response = st.session_state.openai_processor.query_data_with_rag(
                            question,
                            st.session_state.processor.data_summary,
                            st.session_state.processor.vector_store,
                            st.session_state.processor.conversation_manager
                        )

                        # Display results
                        st.subheader("🎯 RAG-Enhanced Analysis Results")
                        st.write(response)

                        # Show retrieved context if requested
                        if show_retrieved_docs:
                            with st.expander("📚 Retrieved Context Documents"):
                                retrieved_docs = st.session_state.processor.vector_store.search(question, k=rag_k_results)
                                for i, doc in enumerate(retrieved_docs, 1):
                                    st.write(f"**Document {i}** (Similarity: {doc['score']:.3f})")
                                    st.write(doc['text'])
                                    if doc.get('metadata'):
                                        st.caption(f"Metadata: {doc['metadata']}")
                                    st.markdown("---")

                        # Show conversation context if requested
                        if show_conversation_context and maintain_context:
                            with st.expander("💬 Conversation Context"):
                                history = st.session_state.processor.conversation_manager.conversation_history
                                for i, exchange in enumerate(reversed(history[-3:]), 1):
                                    st.write(f"**Exchange {len(history) - i + 1}:**")
                                    st.write(f"*Q:* {exchange['question'][:150]}...")
                                    st.write(f"*A:* {exchange['answer'][:200]}...")
                                    st.markdown("---")

                # RAG Testing and Exploration
                st.header("🔍 RAG System Testing & Exploration")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Vector Search Test")
                    test_query = st.text_input("Test semantic search:",
                                             placeholder="e.g., numeric columns with high variance")

                    if test_query and st.button("🔍 Search Documents"):
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
                st.header("⚠️ RAG System Not Ready")
                st.warning("Please build the RAG index to enable enhanced querying with retrieval capabilities.")

                # Fallback to basic querying
                st.subheader("🤖 Basic Natural Language Queries")
                st.markdown("*Note: Using basic mode without RAG enhancement*")

                question = st.text_area("Ask a basic question about your data:",
                                       placeholder="e.g., What are the main patterns in this dataset?")

                if st.button("🔍 Basic Query") and question:
                    with st.spinner("Processing basic query..."):
                        # Use legacy query method without RAG
                        basic_prompt = f"""You are a data analyst. Analyze this dataset:

Dataset: {st.session_state.processor.data_summary['dataset_info']['total_rows']} rows × {st.session_state.processor.data_summary['dataset_info']['total_columns']} columns
Columns: {', '.join(st.session_state.processor.data_summary['dataset_info']['column_names'][:10])}

Question: {question}"""

                        try:
                            response = st.session_state.openai_processor.client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[{"role": "user", "content": basic_prompt}],
                                max_tokens=800,
                                temperature=0.7
                            )
                            st.write(response.choices[0].message.content)
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

# Export functionality
            if st.session_state.processor.tokenized_df is not None:
                st.header("📥 Export Enhanced Data & RAG Components")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("📊 Download Processed Data"):
                        csv = st.session_state.processor.processed_df.to_csv(index=False)
                        st.download_button(
                            label="Download Processed CSV",
                            data=csv,
                            file_name="processed_data.csv",
                            mime="text/csv"
                        )

                with col2:
                    if st.button("🏷️ Download Tokenized Data"):
                        csv = st.session_state.processor.tokenized_df.to_csv(index=False)
                        st.download_button(
                            label="Download Tokenized CSV",
                            data=csv,
                            file_name="tokenized_data.csv",
                            mime="text/csv"
                        )

                with col3:
                    if st.button("🧠 Download RAG Documents"):
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

                # Export comprehensive analysis report
                if st.session_state.processor.tokenization_summary:
                    if st.button("📋 Download Complete Analysis Report"):
                        # Create comprehensive report
                        report = {
                            "version": "2.0",
                            "timestamp": datetime.now().isoformat(),
                            "dataset_summary": st.session_state.processor.data_summary,
                            "tokenization_summary": st.session_state.processor.tokenization_summary,
                            "rag_info": {
                                "documents_count": len(st.session_state.processor.vector_store.documents),
                                "vector_dimension": st.session_state.processor.vector_store.dimension,
                                "rag_ready": st.session_state.rag_ready
                            },
                            "conversation_history": st.session_state.processor.conversation_manager.conversation_history
                        }

                        report_json = json.dumps(report, indent=2)
                        st.download_button(
                            label="Download Complete Analysis Report JSON",
                            data=report_json,
                            file_name="complete_analysis_report_v2.json",
                            mime="application/json"
                        )

    # Footer with version 2.0 features
    st.markdown("---")
    st.markdown("""
    ## 🚀 Version 2.0 - RAG-Enhanced Features

    **What's New in v2.0:**
    - **🧠 RAG (Retrieval-Augmented Generation)**: Vector-based document retrieval for more accurate responses
    - **💬 Conversation Context**: Maintains context across multiple queries for coherent discussions
    - **📚 Semantic Document Search**: FAISS-powered similarity search through your data
    - **🔍 Enhanced Query Processing**: Combines traditional tokenization with modern RAG techniques
    - **📊 Intelligent Context Management**: Smart token management with conversation history

    **Advanced Tokenization Features (from v1.0):**
    - **Contextual Numeric Tokens**: Numbers include magnitude, sign, and range information
    - **Temporal Intelligence**: Dates are tokenized with seasons, decades, and relative time context
    - **Semantic Categories**: Text categories are broken down into meaningful semantic components
    - **Frequency Context**: Common vs rare values are identified and tokenized accordingly
    - **Column Awareness**: All tokens include column context for better querying

    **Best Practices for v2.0:**
    - Build RAG index for enhanced query capabilities
    - Use conversational queries to build on previous insights
    - Leverage vector search to find relevant data patterns
    - Ask follow-up questions to maintain context
    - Use specific questions to get the most from RAG retrieval

    💡 **Pro Tip**: The RAG system excels at connecting related concepts across your dataset. Ask broad questions first, then drill down with follow-ups!

    **Required Dependencies for Full Functionality:**
    ```bash
    pip install streamlit pandas numpy openai plotly nltk tiktoken faiss-cpu sentence-transformers
    ```

    **Installation Commands:**
    ```bash
    # Core dependencies (required)
    pip install streamlit pandas numpy openai plotly nltk tiktoken

    # RAG enhancements (optional but recommended for full features)
    pip install faiss-cpu sentence-transformers
    ```

    **How to Run:**
    ```bash
    # Save this code as csv_rag_app_v2.py, then run:
    streamlit run csv_rag_app_v2.py
    ```
    """)

if __name__ == "__main__":
    main()

