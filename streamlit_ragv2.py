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

# RAG-specific imports with better dependency checking
try:
    from usearch.index import Index
    USEARCH_AVAILABLE = True
except ImportError:
    USEARCH_AVAILABLE = False

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
    page_title="Job Data RAG Analyzer v3.0",
    page_icon="💼",
    layout="wide"
)


class RAGVectorStore:
    """Fixed vector store for RAG functionality optimized for job data"""
    
    def __init__(self):
        self.embedder = None
        self.index = None
        self.documents = []
        self.metadata = []
        self.dimension = 384
        self._index_built = False
        self._initialization_error = None
        
        # Check dependencies
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self._initialization_error = "SentenceTransformers not available"
            st.error("Install SentenceTransformers: pip install sentence-transformers")
            return
            
        if not USEARCH_AVAILABLE:
            self._initialization_error = "USearch not available"
            st.error("Install USearch: pip install usearch")
            return
        
        # Initialize sentence transformer
        try:
            with st.spinner("Loading sentence transformer model..."):
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                test_embedding = self.embedder.encode(["test"], show_progress_bar=False)
                self.dimension = len(test_embedding[0])
                st.success("Sentence transformer loaded successfully")
                logger.info(f"Loaded sentence transformer with dimension: {self.dimension}")
        except Exception as e:
            self._initialization_error = f"Sentence transformer error: {str(e)}"
            st.error(f"Error loading sentence transformer: {str(e)}")
            return
            
        # Initialize USearch index
        try:
            self._init_index()
            st.success("USearch index initialized successfully")
        except Exception as e:
            self._initialization_error = f"USearch error: {str(e)}"
            st.error(f"Error initializing USearch index: {str(e)}")
    
    def _init_index(self):
        """Initialize the USearch index with proper data types"""
        self.index = Index(
            ndim=self.dimension,
            metric='cos',
            dtype=np.float32
        )
    
    def is_available(self) -> bool:
        """Check if RAG functionality is available"""
        return (self.embedder is not None and 
                self.index is not None and 
                self._initialization_error is None)
    
    def build_index(self, chunks: List[Dict]) -> bool:
        """Build USearch index from job data chunks"""
        if not self.is_available():
            st.error("RAG system not available. Install required dependencies.")
            return False
            
        if not chunks:
            st.error("No document chunks provided for indexing")
            return False
            
        try:
            # Clear existing data
            self.documents = []
            self.metadata = []
            self._index_built = False
            self._init_index()
            
            # Extract texts for embedding
            texts = [chunk['text'] for chunk in chunks]
            
            st.info(f"Building vector index for {len(texts)} job data documents...")
            progress_bar = st.progress(0)
            
            # Process in smaller batches for stability
            batch_size = 5
            total_processed = 0
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_chunks = chunks[i:i + batch_size]
                
                try:
                    # Generate embeddings
                    embeddings = self.embedder.encode(
                        batch_texts, 
                        convert_to_tensor=False, 
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
                    
                    if not isinstance(embeddings, np.ndarray):
                        embeddings = np.array(embeddings)
                    
                    embeddings = embeddings.astype(np.float32)
                    
                    # Add each embedding to index
                    for j, embedding in enumerate(embeddings):
                        doc_id = total_processed + j
                        try:
                            embedding_vector = embedding.flatten().astype(np.float32)
                            
                            if len(embedding_vector) != self.dimension:
                                logger.error(f"Embedding dimension mismatch: {len(embedding_vector)} vs {self.dimension}")
                                continue
                            
                            self.index.add(doc_id, embedding_vector)
                            self.documents.append(batch_texts[j])
                            self.metadata.append(batch_chunks[j].get('metadata', {}))
                            
                        except Exception as add_error:
                            logger.error(f"Error adding document {doc_id}: {str(add_error)}")
                            continue
                    
                    total_processed += len(embeddings)
                    progress_bar.progress(min(1.0, total_processed / len(texts)))
                    time.sleep(0.05)
                    
                except Exception as batch_error:
                    st.error(f"Error processing batch: {str(batch_error)}")
                    continue
            
            if total_processed > 0:
                self._index_built = True
                st.success(f"Vector index built with {len(self.documents)} documents")
                return True
            else:
                st.error("No documents were successfully indexed")
                return False
            
        except Exception as e:
            st.error(f"Error building vector index: {str(e)}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant documents using vector similarity"""
        if not self.is_available() or not self._index_built:
            return []
            
        try:
            # Generate query embedding
            query_embedding = self.embedder.encode(
                [query], 
                convert_to_tensor=False, 
                show_progress_bar=False,
                normalize_embeddings=True
            )
            
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding[0])
            else:
                query_embedding = query_embedding[0]
            
            query_embedding = query_embedding.astype(np.float32).flatten()
            
            if len(query_embedding) != self.dimension:
                logger.error(f"Query embedding dimension mismatch")
                return []
            
            # Perform search
            search_results = self.index.search(query_embedding, k)
            
            results = []
            
            # Handle different USearch result formats
            if hasattr(search_results, 'keys') and hasattr(search_results, 'distances'):
                doc_ids = search_results.keys
                distances = search_results.distances
            elif isinstance(search_results, tuple) and len(search_results) == 2:
                doc_ids, distances = search_results
            else:
                doc_ids = getattr(search_results, 'keys', [])
                distances = getattr(search_results, 'distances', [])
            
            # Convert to lists if needed
            if hasattr(doc_ids, 'tolist'):
                doc_ids = doc_ids.tolist()
            if hasattr(distances, 'tolist'):
                distances = distances.tolist()
            
            # Process results
            for i, (doc_id, distance) in enumerate(zip(doc_ids, distances)):
                if isinstance(doc_id, (list, np.ndarray)):
                    doc_id = doc_id[0] if len(doc_id) > 0 else 0
                if isinstance(distance, (list, np.ndarray)):
                    distance = distance[0] if len(distance) > 0 else 1.0
                    
                doc_id = int(doc_id)
                distance = float(distance)
                
                if 0 <= doc_id < len(self.documents):
                    similarity_score = max(0.0, 1.0 - (distance / 2.0))
                    
                    results.append({
                        'text': self.documents[doc_id],
                        'score': similarity_score,
                        'rank': i + 1,
                        'metadata': self.metadata[doc_id] if doc_id < len(self.metadata) else {},
                        'doc_id': doc_id,
                        'distance': distance
                    })
            
            # Sort by score
            results.sort(key=lambda x: x['score'], reverse=True)
            return results
            
        except Exception as e:
            st.error(f"Error during vector search: {str(e)}")
            return []
    
    def create_document_chunks(self, processed_df: pd.DataFrame, data_summary: Dict) -> List[Dict]:
        """Create optimized document chunks from job data"""
        if not self.is_available():
            return []
            
        chunks = []
        
        try:
            # 1. Create row-level job documents (main content)
            for idx, row in processed_df.iterrows():
                # Create comprehensive job description
                job_text_parts = []
                
                # Basic job info
                if pd.notna(row.get('company')):
                    job_text_parts.append(f"Company: {row['company']}")
                
                if pd.notna(row.get('summary_job_title')):
                    job_text_parts.append(f"Job Title: {row['summary_job_title']}")
                elif pd.notna(row.get('displayed_job_title')):
                    job_text_parts.append(f"Job Title: {row['displayed_job_title']}")
                
                # Location information
                location_parts = []
                if pd.notna(row.get('city_job_location')):
                    location_parts.append(row['city_job_location'])
                if pd.notna(row.get('state_job_location')):
                    location_parts.append(row['state_job_location'])
                if pd.notna(row.get('country_job_location')):
                    location_parts.append(row['country_job_location'])
                
                if location_parts:
                    job_text_parts.append(f"Location: {', '.join(location_parts)}")
                
                # Job description (truncated if too long)
                if pd.notna(row.get('job_description')):
                    desc = str(row['job_description'])[:500]  # Limit description length
                    job_text_parts.append(f"Description: {desc}")
                
                # Salary information
                if pd.notna(row.get('job_salary')):
                    job_text_parts.append(f"Salary: {row['job_salary']}")
                
                # Date information
                if pd.notna(row.get('date')):
                    job_text_parts.append(f"Date: {row['date']}")
                
                job_text = ". ".join(job_text_parts)
                
                chunks.append({
                    'text': job_text,
                    'type': 'job_listing',
                    'job_id': idx,
                    'metadata': {
                        'company': str(row.get('company', 'Unknown')),
                        'title': str(row.get('summary_job_title', row.get('displayed_job_title', 'Unknown'))),
                        'location': ', '.join(location_parts) if location_parts else 'Unknown',
                        'row_index': idx
                    }
                })
            
            # 2. Create company-level aggregations
            if 'company' in processed_df.columns:
                company_groups = processed_df.groupby('company')
                for company, group in company_groups:
                    if pd.notna(company) and company != 'Unknown':
                        job_count = len(group)
                        titles = group['summary_job_title'].dropna().unique()[:5]  # Top 5 titles
                        locations = group['city_job_location'].dropna().unique()[:3]  # Top 3 locations
                        
                        company_text = f"Company {company} has {job_count} job listings"
                        if len(titles) > 0:
                            company_text += f" for positions: {', '.join(titles)}"
                        if len(locations) > 0:
                            company_text += f" in locations: {', '.join(locations)}"
                        
                        chunks.append({
                            'text': company_text,
                            'type': 'company_summary',
                            'company': company,
                            'metadata': {
                                'job_count': job_count,
                                'company_name': company
                            }
                        })
            
            # 3. Create location-based aggregations
            if 'city_job_location' in processed_df.columns:
                location_groups = processed_df.groupby('city_job_location')
                for location, group in location_groups:
                    if pd.notna(location) and location != 'Unknown':
                        job_count = len(group)
                        companies = group['company'].dropna().unique()[:3]
                        titles = group['summary_job_title'].dropna().unique()[:3]
                        
                        location_text = f"Location {location} has {job_count} job opportunities"
                        if len(companies) > 0:
                            location_text += f" from companies: {', '.join(companies)}"
                        if len(titles) > 0:
                            location_text += f" for roles: {', '.join(titles)}"
                        
                        chunks.append({
                            'text': location_text,
                            'type': 'location_summary',
                            'location': location,
                            'metadata': {
                                'job_count': job_count,
                                'location_name': location
                            }
                        })
            
            # 4. Create dataset overview
            total_jobs = len(processed_df)
            unique_companies = processed_df['company'].nunique() if 'company' in processed_df.columns else 0
            unique_locations = processed_df['city_job_location'].nunique() if 'city_job_location' in processed_df.columns else 0
            
            overview_text = f"Job dataset contains {total_jobs} total job listings"
            if unique_companies > 0:
                overview_text += f" from {unique_companies} different companies"
            if unique_locations > 0:
                overview_text += f" across {unique_locations} different cities"
            
            chunks.append({
                'text': overview_text,
                'type': 'dataset_overview',
                'metadata': {
                    'total_jobs': total_jobs,
                    'unique_companies': unique_companies,
                    'unique_locations': unique_locations
                }
            })
            
            st.info(f"Created {len(chunks)} document chunks for job data indexing")
            return chunks
            
        except Exception as e:
            st.error(f"Error creating document chunks: {str(e)}")
            return []
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            'available': self.is_available(),
            'index_built': self._index_built,
            'document_count': len(self.documents),
            'dimension': self.dimension,
            'initialization_error': self._initialization_error
        }


class JobDataTokenizer:
    """Specialized tokenizer for job listing data"""
    
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
        """Enhanced text tokenization for job descriptions"""
        if pd.isna(text) or not isinstance(text, str):
            return []

        if not NLTK_AVAILABLE or not self.stemmer:
            # Fallback tokenization
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            return [token for token in text.split() if len(token) > 2]

        try:
            # Convert to lowercase and clean
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)

            # Tokenize
            tokens = word_tokenize(text)

            # Remove stopwords and short tokens
            tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]

            # Apply stemming or lemmatization
            if method == 'stem' and self.stemmer:
                tokens = [self.stemmer.stem(token) for token in tokens]
            elif method == 'lemmatize' and self.lemmatizer:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

            return [token for token in tokens if token.strip()]
            
        except Exception as e:
            logger.error(f"Error in text tokenization: {str(e)}")
            # Fallback
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            return [token for token in text.split() if len(token) > 2]

    def tokenize_job_title(self, title: str) -> List[str]:
        """Specialized tokenization for job titles"""
        if pd.isna(title) or not isinstance(title, str):
            return ['unknown_title']

        tokens = []
        
        try:
            # Basic tokenization
            basic_tokens = self.tokenize_text(title, method='lemmatize')
            tokens.extend(basic_tokens)
            
            # Add title-specific context
            tokens.append('job_title')
            
            # Add seniority level indicators
            title_lower = title.lower()
            if any(word in title_lower for word in ['senior', 'sr', 'lead', 'principal']):
                tokens.extend(['senior_level', 'experienced'])
            elif any(word in title_lower for word in ['junior', 'jr', 'entry', 'associate']):
                tokens.extend(['junior_level', 'entry_level'])
            elif any(word in title_lower for word in ['manager', 'director', 'head', 'chief']):
                tokens.extend(['management', 'leadership'])
            
            # Add domain indicators
            if any(word in title_lower for word in ['engineer', 'developer', 'programmer']):
                tokens.extend(['technical', 'engineering'])
            elif any(word in title_lower for word in ['analyst', 'data', 'research']):
                tokens.extend(['analytical', 'data_role'])
            elif any(word in title_lower for word in ['sales', 'marketing', 'business']):
                tokens.extend(['business', 'commercial'])
            elif any(word in title_lower for word in ['design', 'creative', 'ui', 'ux']):
                tokens.extend(['creative', 'design'])
                
        except Exception as e:
            logger.error(f"Error tokenizing job title {title}: {str(e)}")
            tokens = ['job_title', 'unknown']

        return tokens

    def tokenize_company(self, company: str) -> List[str]:
        """Tokenize company names with context"""
        if pd.isna(company) or not isinstance(company, str):
            return ['unknown_company']

        tokens = []
        
        try:
            # Basic tokenization
            basic_tokens = self.tokenize_text(company, method='lemmatize')
            tokens.extend(basic_tokens)
            
            # Add company context
            tokens.append('company_name')
            
            # Add original cleaned name
            clean_company = re.sub(r'[^\w\s]', '_', company.lower())
            tokens.append(clean_company)
            
        except Exception as e:
            logger.error(f"Error tokenizing company {company}: {str(e)}")
            tokens = ['company_name', 'unknown']

        return tokens

    def tokenize_location(self, location: str, location_type: str = 'city') -> List[str]:
        """Tokenize location data with geographic context"""
        if pd.isna(location) or not isinstance(location, str):
            return [f'unknown_{location_type}']

        tokens = []
        
        try:
            # Basic tokenization
            basic_tokens = self.tokenize_text(location, method='lemmatize')
            tokens.extend(basic_tokens)
            
            # Add location context
            tokens.append(f'{location_type}_location')
            
            # Add cleaned location name
            clean_location = re.sub(r'[^\w\s]', '_', location.lower())
            tokens.append(clean_location)
            
            # Add geographic context based on common patterns
            location_lower = location.lower()
            if location_type == 'city':
                # Add major city indicators
                major_cities = ['new york', 'los angeles', 'chicago', 'houston', 'phoenix', 
                              'philadelphia', 'san antonio', 'san diego', 'dallas', 'san jose',
                              'austin', 'jacksonville', 'fort worth', 'columbus', 'charlotte']
                if any(city in location_lower for city in major_cities):
                    tokens.append('major_city')
                    
        except Exception as e:
            logger.error(f"Error tokenizing location {location}: {str(e)}")
            tokens = [f'{location_type}_location', 'unknown']

        return tokens

    def tokenize_salary(self, salary: str) -> List[str]:
        """Tokenize salary information with range context"""
        if pd.isna(salary) or not isinstance(salary, str):
            return ['salary_unknown']

        tokens = []
        
        try:
            # Add basic salary context
            tokens.append('salary_info')
            
            # Extract numeric values
            numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', str(salary))
            
            if numbers:
                # Convert to numeric and categorize
                try:
                    max_salary = max([float(num.replace(',', '')) for num in numbers])
                    
                    if max_salary < 40000:
                        tokens.extend(['low_salary', 'entry_pay'])
                    elif max_salary < 80000:
                        tokens.extend(['medium_salary', 'mid_range_pay'])
                    elif max_salary < 120000:
                        tokens.extend(['high_salary', 'senior_pay'])
                    else:
                        tokens.extend(['very_high_salary', 'executive_pay'])
                        
                except ValueError:
                    tokens.append('salary_numeric_error')
            
            # Check for salary type indicators
            salary_lower = salary.lower()
            if 'hour' in salary_lower:
                tokens.append('hourly_rate')
            elif any(word in salary_lower for word in ['year', 'annual', 'yearly']):
                tokens.append('annual_salary')
            elif any(word in salary_lower for word in ['month', 'monthly']):
                tokens.append('monthly_salary')
                
        except Exception as e:
            logger.error(f"Error tokenizing salary {salary}: {str(e)}")
            tokens = ['salary_info', 'error']

        return tokens


class ConversationManager:
    """Enhanced conversation manager for job data queries"""
    
    def __init__(self):
        self.conversation_history = []
        self.max_history_length = 15  # Increased for job data context
        
    def add_exchange(self, question: str, answer: str, context_used: List[str] = None):
        """Add a question-answer exchange to conversation history"""
        exchange = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'context_used': context_used or [],
            'query_type': self._classify_query(question)
        }
        
        self.conversation_history.append(exchange)
        
        # Trim history if too long
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def _classify_query(self, question: str) -> str:
        """Classify the type of job-related query"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['company', 'employer', 'firm']):
            return 'company_query'
        elif any(word in question_lower for word in ['location', 'city', 'state', 'where']):
            return 'location_query'
        elif any(word in question_lower for word in ['salary', 'pay', 'compensation', 'wage']):
            return 'salary_query'
        elif any(word in question_lower for word in ['title', 'position', 'role', 'job']):
            return 'job_title_query'
        elif any(word in question_lower for word in ['skill', 'requirement', 'qualification']):
            return 'skills_query'
        elif any(word in question_lower for word in ['summary', 'overview', 'analyze', 'insight']):
            return 'analysis_query'
        else:
            return 'general_query'
    
    def get_context_for_query(self, current_question: str) -> str:
        """Get relevant conversation context for job queries"""
        if not self.conversation_history:
            return ""
        
        current_type = self._classify_query(current_question)
        
        # Get recent relevant exchanges
        relevant_history = []
        for exchange in reversed(self.conversation_history[-5:]):
            if exchange['query_type'] == current_type or exchange['query_type'] == 'analysis_query':
                relevant_history.append(exchange)
        
        if not relevant_history:
            # Fallback to recent history
            relevant_history = self.conversation_history[-3:]
        
        context_parts = ["Previous relevant conversation:"]
        for i, exchange in enumerate(relevant_history[:3], 1):
            context_parts.append(f"Q{i}: {exchange['question'][:100]}...")
            context_parts.append(f"A{i}: {exchange['answer'][:150]}...")
        
        return "\n".join(context_parts)
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


class OpenAIQueryProcessor:
    """Enhanced OpenAI processor optimized for job data analysis"""
    
    def __init__(self, api_key: str):
        self._api_key = api_key
        self.client = None
        self.tokenizer = None
        self.max_context_length = 16385
        self.max_completion_tokens = 2000  # Increased for job analysis
        self.max_input_tokens = self.max_context_length - self.max_completion_tokens - 100
        self._token_cache = {}
        self._initialization_error = None
        
        # Initialize OpenAI client
        try:
            self.client = openai.OpenAI(api_key=api_key)
            # Test connection
            test_response = self.client.models.list()
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            self._initialization_error = f"OpenAI initialization error: {str(e)}"
            st.error(f"Error initializing OpenAI client: {str(e)}")
            return
        
        # Initialize tokenizer for gpt-3.5-turbo
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
                logger.info("Tiktoken tokenizer initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to load tiktoken: {e}")
                try:
                    self.tokenizer = tiktoken.get_encoding("cl100k_base")
                    logger.info("Tiktoken fallback tokenizer initialized")
                except Exception as e2:
                    logger.error(f"Failed to load fallback tiktoken: {e2}")
                    self.tokenizer = None

    def is_available(self) -> bool:
        """Check if OpenAI client is properly initialized"""
        return self.client is not None and self._initialization_error is None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text with caching"""
        if not text:
            return 0
            
        text_hash = hash(text)
        if text_hash in self._token_cache:
            return self._token_cache[text_hash]
            
        if self.tokenizer and TIKTOKEN_AVAILABLE:
            try:
                token_count = len(self.tokenizer.encode(text))
                self._token_cache[text_hash] = token_count
                return token_count
            except Exception as e:
                logger.warning(f"Error counting tokens: {e}")

        # Fallback approximation
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

    def generate_job_system_prompt(self, data_summary: Dict) -> str:
        """Generate system prompt optimized for job data analysis"""
        try:
            dataset_info = data_summary.get('dataset_info', {})
            prompt = f"""You are an expert job market analyst and career advisor with access to a comprehensive job dataset.

Dataset Overview: {dataset_info.get('total_rows', 0):,} job listings across {dataset_info.get('total_columns', 0)} data fields
Key Fields: Company, Job Title, Location, Salary, Job Description, Date

Your expertise includes:
- Job market analysis and trends
- Salary benchmarking and compensation analysis
- Geographic job distribution insights
- Company hiring patterns
- Career guidance and job search optimization
- Skills and qualifications assessment

You have access to a RAG (Retrieval-Augmented Generation) system that provides relevant job data context for each query. Use this context to provide:

1. **Data-driven insights**: Base responses on actual job listings in the dataset
2. **Specific examples**: Reference actual companies, positions, and locations when relevant
3. **Actionable advice**: Provide practical guidance for job seekers and career development
4. **Market context**: Explain trends and patterns in the job market
5. **Comparative analysis**: Help users understand relative opportunities and competition

Instructions:
- Always use retrieved context to support your analysis
- Provide specific data points and examples when available
- Maintain professional, helpful tone appropriate for career guidance
- Focus on actionable insights that help users make informed decisions
- Acknowledge limitations when dataset doesn't contain relevant information"""

            # Ensure system prompt fits within limits
            max_system_tokens = int(self.max_input_tokens * 0.4)
            return self.truncate_text(prompt, max_system_tokens)
            
        except Exception as e:
            logger.error(f"Error generating system prompt: {str(e)}")
            return "You are a job market analyst. Help analyze the job dataset and provide career insights."

    def prepare_job_rag_context(self, retrieved_docs: List[Dict], conversation_context: str = "") -> str:
        """Prepare RAG context optimized for job data"""
        context_parts = []
        
        try:
            if conversation_context:
                context_parts.append(f"Previous Conversation:\n{conversation_context}")
            
            if retrieved_docs:
                context_parts.append("Relevant Job Data:")
                
                # Group documents by type for better organization
                job_listings = [doc for doc in retrieved_docs if doc.get('metadata', {}).get('company')]
                company_summaries = [doc for doc in retrieved_docs if 'company_summary' in doc.get('text', '')]
                location_summaries = [doc for doc in retrieved_docs if 'location' in doc.get('text', '')]
                other_docs = [doc for doc in retrieved_docs if doc not in job_listings + company_summaries + location_summaries]
                
                # Add job listings first (most important)
                if job_listings:
                    context_parts.append("\nSpecific Job Listings:")
                    for i, doc in enumerate(job_listings[:3], 1):
                        score = doc.get('score', 0)
                        metadata = doc.get('metadata', {})
                        context_parts.append(f"{i}. [Score: {score:.3f}] {doc.get('text', '')}")
                        if metadata.get('company'):
                            context_parts.append(f"   Company: {metadata['company']}, Location: {metadata.get('location', 'N/A')}")
                
                # Add company summaries
                if company_summaries:
                    context_parts.append("\nCompany Information:")
                    for doc in company_summaries[:2]:
                        context_parts.append(f"• {doc.get('text', '')}")
                
                # Add location summaries
                if location_summaries:
                    context_parts.append("\nLocation Insights:")
                    for doc in location_summaries[:2]:
                        context_parts.append(f"• {doc.get('text', '')}")
                
                # Add other relevant information
                if other_docs:
                    context_parts.append("\nAdditional Context:")
                    for doc in other_docs[:2]:
                        context_parts.append(f"• {doc.get('text', '')}")
            
            full_context = "\n".join(context_parts)
            
            # Reserve space for RAG context
            max_context_tokens = int(self.max_input_tokens * 0.5)
            return self.truncate_text(full_context, max_context_tokens)
            
        except Exception as e:
            logger.error(f"Error preparing RAG context: {str(e)}")
            return ""

    def query_job_data_with_rag(self, question: str, data_summary: Dict, vector_store: RAGVectorStore, 
                               conversation_manager: ConversationManager) -> str:
        """Process job-related queries with RAG enhancement"""
        
        if not self.is_available():
            return f"OpenAI client not available: {self._initialization_error}"
        
        try:
            # Get conversation context
            conversation_context = conversation_manager.get_context_for_query(question)
            
            # Retrieve relevant job documents
            retrieved_docs = []
            try:
                retrieved_docs = vector_store.search(question, k=8)  # More docs for job analysis
                logger.info(f"Retrieved {len(retrieved_docs)} job documents for query")
            except Exception as search_error:
                logger.warning(f"Vector search failed: {search_error}")
            
            # Generate job-specific system prompt
            system_prompt = self.generate_job_system_prompt(data_summary)
            
            # Prepare job-specific RAG context
            rag_context = self.prepare_job_rag_context(retrieved_docs, conversation_context)
            
            # Create user message
            user_message = f"Question: {question}"
            if rag_context:
                user_message += f"\n\nRelevant Job Data Context:\n{rag_context}"
            
            # Token management
            system_tokens = self.count_tokens(system_prompt)
            user_tokens = self.count_tokens(user_message)
            
            total_input_tokens = system_tokens + user_tokens
            
            if total_input_tokens > self.max_input_tokens:
                excess_tokens = total_input_tokens - self.max_input_tokens
                if rag_context:
                    current_context_tokens = self.count_tokens(rag_context)
                    reduced_context_tokens = max(300, current_context_tokens - excess_tokens)
                    rag_context = self.truncate_text(rag_context, reduced_context_tokens)
                    user_message = f"Question: {question}\n\nRelevant Job Data Context:\n{rag_context}"
            
            # Calculate available completion tokens
            final_input_tokens = self.count_tokens(system_prompt) + self.count_tokens(user_message)
            available_tokens = self.max_context_length - final_input_tokens - 100
            completion_tokens = min(self.max_completion_tokens, max(500, available_tokens))
            
            # Make API call
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=completion_tokens,
                    temperature=0.7,
                    timeout=30
                )
                
                answer = response.choices[0].message.content
                
                # Store in conversation history
                context_used = [doc['text'][:100] + "..." for doc in retrieved_docs[:3]]
                conversation_manager.add_exchange(question, answer, context_used)
                
                logger.info(f"Successfully processed job query: {question[:50]}...")
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
            logger.error(f"Error in job RAG query processing: {error_msg}")
            
            if "maximum context length" in error_msg.lower():
                return """The query is too complex for a single analysis. Please try:
1. Ask more specific questions about particular companies or locations
2. Focus on specific aspects like salary ranges or job titles
3. Break down your analysis into smaller, focused questions

The job dataset has been successfully processed with RAG capabilities - you can explore it using more targeted queries."""
            else:
                return f"Error processing query: {error_msg}. Please try a simpler question or check your OpenAI API key."

class JobCSVProcessor:
    """Enhanced CSV processor specifically optimized for job listing data"""
    
    def __init__(self):
        self.df = None
        self.processed_df = None
        self.tokenized_df = None
        self.data_summary = None
        self.tokenization_summary = None
        self.tokenizer = JobDataTokenizer()
        self.vector_store = RAGVectorStore()
        self.conversation_manager = ConversationManager()

    def load_csv(self, uploaded_file) -> bool:
        """Load and validate job data CSV"""
        try:
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

            # Validate job data structure
            if self.df.empty:
                raise ValueError("CSV file is empty")
                
            if len(self.df.columns) == 0:
                raise ValueError("CSV file has no columns")

            # Check for expected job data columns
            expected_columns = ['company', 'job', 'title', 'description', 'location', 'salary']
            found_columns = []
            
            for col in self.df.columns:
                col_lower = col.lower().replace(' ', '_')
                for expected in expected_columns:
                    if expected in col_lower:
                        found_columns.append(expected)
                        break

            if len(found_columns) < 3:
                st.warning("This doesn't appear to be a typical job dataset. Proceeding with general processing.")
            else:
                st.success(f"Detected job dataset with columns: {', '.join(found_columns)}")

            logger.info(f"Loaded job CSV: {len(self.df)} rows, {len(self.df.columns)} columns")
            return True
            
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            logger.error(f"CSV loading error: {str(e)}")
            return False
            
def clean_job_data(self) -> bool:
        """Clean and preprocess job data with domain-specific logic"""
        if self.df is None:
            st.error("No data loaded")
            return False

        try:
            self.processed_df = self.df.copy()

            # Standardize column names for job data
            column_mapping = {}
            used_names = set()
            
            for col in self.processed_df.columns:
                original_col = col
                clean_col = re.sub(r'[^\w\s]', '', str(col)).strip().replace(' ', '_').lower()
                
                # Map to standard job data column names based on your CSV structure
                if any(word in clean_col for word in ['company', 'employer', 'firm']):
                    clean_col = 'company'
                elif 'summary' in clean_col and 'job' in clean_col and 'title' in clean_col:
                    clean_col = 'summary_job_title'
                elif 'displayed' in clean_col and 'job' in clean_col and 'title' in clean_col:
                    clean_col = 'displayed_job_title'
                elif 'job' in clean_col and 'description' in clean_col:
                    clean_col = 'job_description'
                elif 'detailed' in clean_col and 'job' in clean_col and 'location' in clean_col:
                    clean_col = 'detailed_job_location'
                elif 'city' in clean_col and 'job' in clean_col and 'location' in clean_col:
                    clean_col = 'city_job_location'
                elif 'state' in clean_col and 'job' in clean_col and 'location' in clean_col:
                    clean_col = 'state_job_location'
                elif 'country' in clean_col and 'job' in clean_col and 'location' in clean_col:
                    clean_col = 'country_job_location'
                elif 'job' in clean_col and 'salary' in clean_col:
                    clean_col = 'job_salary'
                elif 'source' in clean_col:
                    clean_col = 'source'
                elif 'date' in clean_col:
                    clean_col = 'date'
                elif 'id' in clean_col:
                    clean_col = 'id'
                
                # Ensure unique column names
                if clean_col in used_names:
                    counter = 1
                    base_name = clean_col
                    while clean_col in used_names:
                        clean_col = f"{base_name}_{counter}"
                        counter += 1
                
                used_names.add(clean_col)
                column_mapping[original_col] = clean_col

            # Apply column name mapping
            self.processed_df = self.processed_df.rename(columns=column_mapping)

            # Handle missing values with job-specific logic
            for col in self.processed_df.columns:
                if col in ['company', 'summary_job_title', 'displayed_job_title']:
                    self.processed_df[col] = self.processed_df[col].fillna('Unknown')
                elif col in ['city_job_location', 'state_job_location', 'country_job_location', 'detailed_job_location']:
                    self.processed_df[col] = self.processed_df[col].fillna('Unknown')
                elif col == 'job_salary':
                    self.processed_df[col] = self.processed_df[col].fillna('Not Specified')
                elif col == 'job_description':
                    self.processed_df[col] = self.processed_df[col].fillna('No description provided')
                elif col == 'source':
                    self.processed_df[col] = self.processed_df[col].fillna('Unknown Source')
                elif self.processed_df[col].dtype == 'object':
                    self.processed_df[col] = self.processed_df[col].fillna('Unknown')
                else:
                    # For numeric columns
                    try:
                        median_val = self.processed_df[col].median()
                        fill_value = median_val if pd.notna(median_val) else 0
                        self.processed_df[col] = self.processed_df[col].fillna(fill_value)
                    except Exception as e:
                        logger.warning(f"Could not calculate median for column {col}: {e}")
                        self.processed_df[col] = self.processed_df[col].fillna(0)

            # Clean text fields specifically - THIS IS THE KEY FIX
            text_columns = ['company', 'summary_job_title', 'displayed_job_title', 'job_description', 
                          'city_job_location', 'state_job_location', 'country_job_location', 
                          'detailed_job_location', 'source']
            
            for col in text_columns:
                if col in self.processed_df.columns:
                    try:
                        # Create a copy of the series to work with
                        series_to_clean = self.processed_df[col].copy()
                        
                        # Ensure it's string type
                        series_to_clean = series_to_clean.astype(str)
                        
                        # Apply string operations safely
                        # Remove extra whitespace
                        series_to_clean = series_to_clean.apply(lambda x: str(x).strip() if pd.notna(x) else 'Unknown')
                        
                        # Replace multiple spaces with single space  
                        series_to_clean = series_to_clean.apply(lambda x: re.sub(r'\s+', ' ', str(x)) if pd.notna(x) else 'Unknown')
                        
                        # Assign back to the DataFrame
                        self.processed_df[col] = series_to_clean
                        
                    except Exception as e:
                        logger.warning(f"Could not clean text column {col}: {e}")
                        try:
                            # Ultimate fallback: just ensure it's string
                            self.processed_df[col] = self.processed_df[col].astype(str)
                        except Exception as e2:
                            logger.error(f"Complete failure cleaning column {col}: {e2}")
                            continue

            # Handle date column if present
            if 'date' in self.processed_df.columns:
                try:
                    # Convert date column
                    self.processed_df['date'] = pd.to_datetime(self.processed_df['date'], errors='coerce')
                    # Fill failed conversions with a default date
                    self.processed_df['date'] = self.processed_df['date'].fillna(pd.Timestamp('2023-01-01'))
                except Exception as e:
                    logger.warning(f"Could not convert date column: {e}")

            # Clean salary information
            if 'job_salary' in self.processed_df.columns:
                try:
                    # Clean salary using apply method to avoid str accessor issues
                    self.processed_df['job_salary'] = self.processed_df['job_salary'].apply(
                        lambda x: re.sub(r'[\$,]', '', str(x)) if pd.notna(x) else 'Not Specified'
                    )
                except Exception as e:
                    logger.warning(f"Could not clean salary column: {e}")
                    # Fallback: just convert to string
                    self.processed_df['job_salary'] = self.processed_df['job_salary'].astype(str)

            logger.info(f"Job data cleaning completed: {len(self.processed_df)} rows, {len(self.processed_df.columns)} columns")
            return True
            
        except Exception as e:
            st.error(f"Error cleaning job data: {str(e)}")
            logger.error(f"Job data cleaning error: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def tokenize_job_dataset(self) -> bool:
        """Perform job-specific tokenization of the dataset"""
        if self.processed_df is None:
            st.error("No processed data available")
            return False

        try:
            st.info("Starting job-specific tokenization process...")

            # Initialize tokenized dataframe
            self.tokenized_df = self.processed_df.copy()

            # Dictionary to store tokenization results
            all_tokens = {}
            column_token_stats = {}

            progress_bar = st.progress(0)
            total_columns = len(self.processed_df.columns)

            for idx, col in enumerate(self.processed_df.columns):
                st.info(f"Tokenizing column: {col}")

                column_tokens = []
                token_column_name = f"{col}_tokens"

                try:
                    # Job-specific tokenization based on column type
                    if col == 'company':
                        # Company name tokenization
                        for value in self.processed_df[col]:
                            tokens = self.tokenizer.tokenize_company(value)
                            column_tokens.extend(tokens)

                    elif col in ['summary_job_title', 'displayed_job_title']:
                        # Job title tokenization
                        for value in self.processed_df[col]:
                            tokens = self.tokenizer.tokenize_job_title(value)
                            column_tokens.extend(tokens)

                    elif col in ['city_job_location', 'state_job_location', 'country_job_location']:
                        # Location tokenization
                        location_type = col.split('_')[0]  # Extract 'city', 'state', or 'country'
                        for value in self.processed_df[col]:
                            tokens = self.tokenizer.tokenize_location(value, location_type)
                            column_tokens.extend(tokens)

                    elif col == 'job_salary':
                        # Salary tokenization
                        for value in self.processed_df[col]:
                            tokens = self.tokenizer.tokenize_salary(value)
                            column_tokens.extend(tokens)

                    elif col == 'job_description':
                        # Job description tokenization (text analysis)
                        for value in self.processed_df[col]:
                            tokens = self.tokenizer.tokenize_text(value, method='lemmatize')
                            column_tokens.extend(tokens)

                    elif pd.api.types.is_datetime64_any_dtype(self.processed_df[col]):
                        # Date tokenization (if applicable)
                        for value in self.processed_df[col]:
                            try:
                                date_str = str(value)
                                tokens = self.tokenizer.tokenize_text(date_str)
                                tokens.extend(['date_field', f'{col}_date'])
                                column_tokens.extend(tokens)
                            except:
                                column_tokens.extend(['date_field', 'unknown_date'])

                    else:
                        # General text tokenization for other fields
                        for value in self.processed_df[col]:
                            tokens = self.tokenizer.tokenize_text(str(value), method='lemmatize')
                            column_tokens.extend(tokens)

                    # Store tokens for this column
                    all_tokens[col] = column_tokens

                    # Calculate token statistics
                    unique_tokens = set(column_tokens)
                    column_token_stats[col] = {
                        'total_tokens': len(column_tokens),
                        'unique_tokens': len(unique_tokens),
                        'most_common': Counter(column_tokens).most_common(10),
                        'token_diversity': len(unique_tokens) / len(column_tokens) if column_tokens else 0
                    }

                    # Create tokenized column for display
                    token_lists = []
                    
                    if col == 'company':
                        for value in self.processed_df[col]:
                            tokens = self.tokenizer.tokenize_company(value)
                            token_lists.append(' | '.join(tokens))
                    elif col in ['summary_job_title', 'displayed_job_title']:
                        for value in self.processed_df[col]:
                            tokens = self.tokenizer.tokenize_job_title(value)
                            token_lists.append(' | '.join(tokens))
                    elif col in ['city_job_location', 'state_job_location', 'country_job_location']:
                        location_type = col.split('_')[0]
                        for value in self.processed_df[col]:
                            tokens = self.tokenizer.tokenize_location(value, location_type)
                            token_lists.append(' | '.join(tokens))
                    elif col == 'job_salary':
                        for value in self.processed_df[col]:
                            tokens = self.tokenizer.tokenize_salary(value)
                            token_lists.append(' | '.join(tokens))
                    else:
                        for value in self.processed_df[col]:
                            tokens = self.tokenizer.tokenize_text(str(value), method='lemmatize')
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

            # Create job-specific tokenization summary
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
                },
                'job_specific_insights': {
                    'companies_tokenized': len(all_tokens.get('company', [])),
                    'job_titles_tokenized': len(all_tokens.get('summary_job_title', [])) + len(all_tokens.get('displayed_job_title', [])),
                    'locations_tokenized': len(all_tokens.get('city_job_location', [])) + len(all_tokens.get('state_job_location', [])),
                    'salary_tokens': len(all_tokens.get('job_salary', []))
                }
            }

            st.success("Job-specific tokenization completed successfully!")
            logger.info(f"Job tokenization completed: {len(valid_stats)} successful columns")
            return True
            
        except Exception as e:
            st.error(f"Error during job tokenization: {str(e)}")
            logger.error(f"Job tokenization error: {str(e)}")
            return False
    
    def build_job_rag_index(self) -> bool:
        """Build RAG vector index optimized for job data"""
        if self.processed_df is None or self.tokenized_df is None:
            st.error("Please process and tokenize job data first")
            return False
        
        try:
            st.info("Building job-specific RAG vector index...")
            
            # Generate data summary if not already done
            if self.data_summary is None:
                self.generate_job_data_summary()
            
            # Create job-specific document chunks
            chunks = self.vector_store.create_document_chunks(self.processed_df, self.data_summary)
            
            # Build vector index
            if chunks and self.vector_store.build_index(chunks):
                st.success("Job RAG index built successfully!")
                logger.info("Job RAG index built successfully")
                return True
            else:
                st.error("Failed to build job RAG index")
                return False
                
        except Exception as e:
            st.error(f"Error building job RAG index: {str(e)}")
            logger.error(f"Job RAG index building error: {str(e)}")
            return False

    def generate_job_data_summary(self) -> Optional[Dict]:
        """Generate comprehensive summary optimized for job data"""
        if self.processed_df is None:
            return None

        try:
            # Basic dataset info
            summary = {
                "dataset_info": {
                    "total_rows": len(self.processed_df),
                    "total_columns": len(self.processed_df.columns),
                    "column_names": list(self.processed_df.columns)
                },
                "column_details": {},
                "tokenization_info": self.tokenization_summary if self.tokenization_summary else None,
                "job_market_insights": {}
            }

            # Analyze each column with job-specific insights
            for col in self.processed_df.columns:
                try:
                    col_info = {
                        "data_type": str(self.processed_df[col].dtype),
                        "null_count": int(self.processed_df[col].isnull().sum()),
                        "unique_count": int(self.processed_df[col].nunique())
                    }

                    # Add job-specific analysis
                    if col == 'company':
                        top_companies = self.processed_df[col].value_counts().head(5)
                        col_info.update({
                            "top_companies": {str(k): int(v) for k, v in top_companies.to_dict().items()},
                            "total_unique_companies": int(self.processed_df[col].nunique())
                        })

                    elif col in ['summary_job_title', 'displayed_job_title']:
                        top_titles = self.processed_df[col].value_counts().head(5)
                        col_info.update({
                            "top_job_titles": {str(k): int(v) for k, v in top_titles.to_dict().items()},
                            "total_unique_titles": int(self.processed_df[col].nunique())
                        })

                    elif col in ['city_job_location', 'state_job_location', 'country_job_location']:
                        top_locations = self.processed_df[col].value_counts().head(5)
                        col_info.update({
                            "top_locations": {str(k): int(v) for k, v in top_locations.to_dict().items()},
                            "total_unique_locations": int(self.processed_df[col].nunique())
                        })

                    elif col == 'job_salary':
                        # Analyze salary data
                        salary_series = self.processed_df[col].astype(str)
                        non_empty_salaries = salary_series[salary_series != 'Not Specified']
                        col_info.update({
                            "salary_specified_count": len(non_empty_salaries),
                            "salary_not_specified_count": len(salary_series) - len(non_empty_salaries),
                            "sample_salaries": list(non_empty_salaries.head(5))
                        })

                    elif self.processed_df[col].dtype in ['int64', 'float64']:
                        col_info.update({
                            "min": float(self.processed_df[col].min()),
                            "max": float(self.processed_df[col].max()),
                            "mean": float(self.processed_df[col].mean()),
                            "median": float(self.processed_df[col].median())
                        })

                    # Add tokenization info if available
                    if self.tokenization_summary and col in self.tokenization_summary['column_stats']:
                        col_info['tokenization'] = self.tokenization_summary['column_stats'][col]
                        
                    summary["column_details"][col] = col_info
                    
                except Exception as col_error:
                    logger.warning(f"Error processing column {col}: {str(col_error)}")
                    summary["column_details"][col] = {
                        "data_type": str(self.processed_df[col].dtype),
                        "error": str(col_error)
                    }

            # Generate job market insights
            try:
                insights = {}
                
                # Company distribution
                if 'company' in self.processed_df.columns:
                    company_counts = self.processed_df['company'].value_counts()
                    insights['company_distribution'] = {
                        'total_companies': len(company_counts),
                        'top_hiring_company': company_counts.index[0] if len(company_counts) > 0 else 'Unknown',
                        'max_jobs_by_company': int(company_counts.iloc[0]) if len(company_counts) > 0 else 0
                    }
                
                # Location distribution
                if 'city_job_location' in self.processed_df.columns:
                    location_counts = self.processed_df['city_job_location'].value_counts()
                    insights['location_distribution'] = {
                        'total_cities': len(location_counts),
                        'top_job_city': location_counts.index[0] if len(location_counts) > 0 else 'Unknown',
                        'max_jobs_by_city': int(location_counts.iloc[0]) if len(location_counts) > 0 else 0
                    }
                
                # Job title analysis
                if 'summary_job_title' in self.processed_df.columns:
                    title_counts = self.processed_df['summary_job_title'].value_counts()
                    insights['job_title_distribution'] = {
                        'total_unique_titles': len(title_counts),
                        'most_common_title': title_counts.index[0] if len(title_counts) > 0 else 'Unknown',
                        'max_occurrences': int(title_counts.iloc[0]) if len(title_counts) > 0 else 0
                    }
                
                summary['job_market_insights'] = insights
                
            except Exception as insights_error:
                logger.warning(f"Error generating job market insights: {str(insights_error)}")
                summary['job_market_insights'] = {'error': str(insights_error)}

            self.data_summary = summary
            logger.info("Job data summary generated successfully")
            return summary
            
        except Exception as e:
            st.error(f"Error generating job data summary: {str(e)}")
            logger.error(f"Job data summary generation error: {str(e)}")
            return None

def main():
    st.title("Job Data RAG Analyzer v3.0 - Optimized for Job Listings")
    st.markdown("Upload your job dataset CSV and ask intelligent questions with **RAG (Retrieval-Augmented Generation)** and **job-specific tokenization**!")
    
    # Sidebar status and configuration
    with st.sidebar:
        st.header("System Status")
        
        # Dependency status
        dep_status = []
        if USEARCH_AVAILABLE:
            dep_status.append("✅ USearch (Vector Search)")
        else:
            dep_status.append("❌ USearch - pip install usearch")
            
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            dep_status.append("✅ SentenceTransformers")
        else:
            dep_status.append("❌ SentenceTransformers - pip install sentence-transformers")
            
        if TIKTOKEN_AVAILABLE:
            dep_status.append("✅ tiktoken")
        else:
            dep_status.append("❌ tiktoken - pip install tiktoken")
            
        if NLTK_AVAILABLE:
            dep_status.append("✅ NLTK")
        else:
            dep_status.append("❌ NLTK")
            
        for status in dep_status:
            if "✅" in status:
                st.success(status)
            else:
                st.error(status)

        st.markdown("---")
        st.header("Configuration")

        # OpenAI API Key
        api_key = st.text_input("OpenAI API Key", type="password",
                               help="Enter your OpenAI API key for RAG-enhanced queries")

        st.markdown("---")
        st.header("RAG Settings")
        
        rag_k_results = st.slider("Retrieved Documents", 3, 15, 8, 
                                 help="Number of relevant job documents to retrieve")
        
        maintain_context = st.checkbox("Maintain Conversation Context", value=True,
                                      help="Keep context from previous job-related questions")

    # Check NLTK data
    if not download_nltk_data():
        st.error("Failed to download required NLTK data. Some tokenization features may not work properly.")

    # Initialize session state
    if 'job_processor' not in st.session_state:
        st.session_state.job_processor = JobCSVProcessor()
    if 'openai_processor' not in st.session_state:
        st.session_state.openai_processor = None
    if 'rag_ready' not in st.session_state:
        st.session_state.rag_ready = False

    # Configure OpenAI if API key provided
    if api_key:
        try:
            st.session_state.openai_processor = OpenAIQueryProcessor(api_key)
            if st.session_state.openai_processor.is_available():
                st.sidebar.success("✅ OpenAI API configured")
            else:
                st.sidebar.error(f"❌ OpenAI error: {st.session_state.openai_processor._initialization_error}")
        except Exception as e:
            st.sidebar.error(f"❌ OpenAI error: {str(e)}")
    else:
        st.sidebar.warning("⚠️ Enter OpenAI API key for enhanced analysis")

    # File upload section
    st.header("📁 Upload Job Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file containing job data", type="csv",
                                   help="Expected columns: Company, Job Title, Location, Salary, Description, etc.")

    if uploaded_file is not None:
        # Load and display basic info
        if st.session_state.job_processor.load_csv(uploaded_file):
            st.success("✅ Job dataset loaded successfully!")

            # Display dataset overview
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Jobs", len(st.session_state.job_processor.df))
            with col2:
                st.metric("Data Columns", len(st.session_state.job_processor.df.columns))
            with col3:
                # Try to identify job-specific columns
                job_cols = 0
                for col in st.session_state.job_processor.df.columns:
                    if any(word in col.lower() for word in ['company', 'job', 'title', 'salary', 'location']):
                        job_cols += 1
                st.metric("Job-Related Columns", job_cols)

            # Show column information
            with st.expander("📊 Dataset Column Information"):
                col_info = []
                for col in st.session_state.job_processor.df.columns:
                    col_info.append({
                        'Column': col,
                        'Type': str(st.session_state.job_processor.df[col].dtype),
                        'Non-Null': f"{st.session_state.job_processor.df[col].count():,}",
                        'Unique Values': f"{st.session_state.job_processor.df[col].nunique():,}",
                        'Sample Value': str(st.session_state.job_processor.df[col].iloc[0])[:50] + "..." if len(str(st.session_state.job_processor.df[col].iloc[0])) > 50 else str(st.session_state.job_processor.df[col].iloc[0])
                    })
                st.dataframe(pd.DataFrame(col_info), use_container_width=True)

            # Show sample data
            with st.expander("👀 Sample Job Data"):
                st.dataframe(st.session_state.job_processor.df.head(10))

            # Processing workflow
            st.header("🔧 Data Processing & RAG Setup")
            
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("🧹 Clean Job Data", type="secondary", use_container_width=True):
                    with st.spinner("Cleaning and standardizing job data..."):
                        if st.session_state.job_processor.clean_job_data():
                            st.success("✅ Job data cleaned successfully!")
                            # Show improvements
                            with st.expander("View Cleaned Data Changes"):
                                st.write("**Standardized Column Names:**")
                                for old, new in zip(st.session_state.job_processor.df.columns, 
                                                  st.session_state.job_processor.processed_df.columns):
                                    if old != new:
                                        st.write(f"• {old} → {new}")
                                
                                st.write("**Missing Value Handling:**")
                                null_counts = st.session_state.job_processor.processed_df.isnull().sum()
                                st.write(f"• Total null values: {null_counts.sum()}")
                        else:
                            st.error("❌ Error cleaning job data")

            with col2:
                if st.button("🔤 Tokenize Job Data", type="secondary", use_container_width=True):
                    if st.session_state.job_processor.processed_df is not None:
                        with st.spinner("Performing job-specific tokenization..."):
                            if st.session_state.job_processor.tokenize_job_dataset():
                                st.session_state.job_processor.generate_job_data_summary()
                                st.success("✅ Job tokenization completed!")
                            else:
                                st.error("❌ Error during job tokenization")
                    else:
                        st.error("Please clean data first")

            with col3:
                if st.button("🧠 Build RAG Index", type="primary", use_container_width=True):
                    if (st.session_state.job_processor.processed_df is not None and 
                        st.session_state.job_processor.tokenized_df is not None):
                        with st.spinner("Building job-specific RAG vector index..."):
                            if st.session_state.job_processor.build_job_rag_index():
                                st.session_state.rag_ready = True
                                st.success("✅ Job RAG system ready!")
                            else:
                                st.error("❌ Error building RAG index")
                    else:
                        st.error("Please clean and tokenize data first")

            # One-click workflow
            st.markdown("### 🚀 Complete Workflow")
            if st.button("🔄 Process Everything (Clean + Tokenize + Build RAG)", type="primary", use_container_width=True):
                workflow_success = True
                
                with st.spinner("Running complete job data processing workflow..."):
                    # Step 1: Clean
                    st.info("Step 1/3: Cleaning job data...")
                    if not st.session_state.job_processor.clean_job_data():
                        st.error("❌ Data cleaning failed")
                        workflow_success = False
                    else:
                        st.success("✅ Data cleaned")
                    
                    # Step 2: Tokenize
                    if workflow_success:
                        st.info("Step 2/3: Tokenizing job data...")
                        if st.session_state.job_processor.tokenize_job_dataset():
                            st.session_state.job_processor.generate_job_data_summary()
                            st.success("✅ Data tokenized")
                        else:
                            st.error("❌ Tokenization failed")
                            workflow_success = False
                    
                    # Step 3: Build RAG
                    if workflow_success:
                        st.info("Step 3/3: Building RAG index...")
                        if st.session_state.job_processor.build_job_rag_index():
                            st.session_state.rag_ready = True
                            st.success("✅ Complete RAG system ready!")
                        else:
                            st.error("❌ RAG index building failed")
                            workflow_success = False

                if workflow_success:
                    st.balloons()
                    st.success("🎉 Complete job data processing workflow completed successfully!")

            # Show processing status
            status_cols = st.columns(3)
            with status_cols[0]:
                if st.session_state.job_processor.processed_df is not None:
                    st.info("✅ Data Cleaned")
                else:
                    st.warning("⏳ Data Not Cleaned")
            
            with status_cols[1]:
                if st.session_state.job_processor.tokenized_df is not None:
                    st.info("✅ Data Tokenized")
                else:
                    st.warning("⏳ Data Not Tokenized")
            
            with status_cols[2]:
                if st.session_state.rag_ready:
                    st.info("✅ RAG System Ready")
                else:
                    st.warning("⏳ RAG Not Built")

            # Enhanced analysis section
            if st.session_state.job_processor.tokenized_df is not None:
                st.header("📈 Job Market Analysis & Insights")

                tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔤 Tokenization", "🏢 Companies", "📍 Locations"])

                with tab1:
                    st.subheader("Job Dataset Overview")

                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Job Listings", f"{len(st.session_state.job_processor.processed_df):,}")
                    with col2:
                        if 'company' in st.session_state.job_processor.processed_df.columns:
                            unique_companies = st.session_state.job_processor.processed_df['company'].nunique()
                            st.metric("Unique Companies", f"{unique_companies:,}")
                        else:
                            st.metric("Unique Companies", "N/A")
                    with col3:
                        if 'city_job_location' in st.session_state.job_processor.processed_df.columns:
                            unique_cities = st.session_state.job_processor.processed_df['city_job_location'].nunique()
                            st.metric("Unique Cities", f"{unique_cities:,}")
                        else:
                            st.metric("Unique Cities", "N/A")
                    with col4:
                        if 'summary_job_title' in st.session_state.job_processor.processed_df.columns:
                            unique_titles = st.session_state.job_processor.processed_df['summary_job_title'].nunique()
                            st.metric("Unique Job Titles", f"{unique_titles:,}")
                        else:
                            st.metric("Unique Job Titles", "N/A")

                    # RAG system status
                    if st.session_state.rag_ready:
                        st.success("🧠 RAG System: Active with job-optimized vector search")
                        vector_stats = st.session_state.job_processor.vector_store.get_stats()
                        st.info(f"📚 Vector Index: {vector_stats['document_count']} documents indexed")
                    else:
                        st.warning("🧠 RAG System: Build index to enable intelligent job queries")

                with tab2:
                    st.subheader("Job-Specific Tokenization Results")

                    if st.session_state.job_processor.tokenization_summary:
                        token_summary = st.session_state.job_processor.tokenization_summary

                        # Global metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Tokens", f"{token_summary['global_stats']['total_tokens_generated']:,}")
                        with col2:
                            st.metric("Unique Tokens", f"{token_summary['global_stats']['total_unique_tokens']:,}")
                        with col3:
                            st.metric("Avg Tokens/Column", f"{token_summary['global_stats']['average_tokens_per_column']:.1f}")
                        with col4:
                            st.metric("Token Diversity", f"{token_summary['global_stats']['average_diversity_per_column']:.3f}")

                        # Job-specific insights
                        if 'job_specific_insights' in token_summary:
                            st.write("### Job-Specific Tokenization Insights")
                            insights = token_summary['job_specific_insights']
                            
                            insight_cols = st.columns(4)
                            with insight_cols[0]:
                                st.metric("Company Tokens", f"{insights.get('companies_tokenized', 0):,}")
                            with insight_cols[1]:
                                st.metric("Job Title Tokens", f"{insights.get('job_titles_tokenized', 0):,}")
                            with insight_cols[2]:
                                st.metric("Location Tokens", f"{insights.get('locations_tokenized', 0):,}")
                            with insight_cols[3]:
                                st.metric("Salary Tokens", f"{insights.get('salary_tokens', 0):,}")

                        # Column breakdown
                        st.write("### Tokenization by Column")
                        column_stats_data = []
                        for col, stats in token_summary['column_stats'].items():
                            column_stats_data.append({
                                'Column': col,
                                'Type': 'Job-Specific' if col in ['company', 'summary_job_title', 'city_job_location', 'job_salary'] else 'General',
                                'Total Tokens': f"{stats['total_tokens']:,}",
                                'Unique Tokens': f"{stats['unique_tokens']:,}",
                                'Diversity': f"{stats['token_diversity']:.3f}",
                                'Top Token': stats['most_common'][0][0] if stats['most_common'] else 'N/A'
                            })

                        st.dataframe(pd.DataFrame(column_stats_data), use_container_width=True)

                with tab3:
                    st.subheader("Company Analysis")
                    
                    if 'company' in st.session_state.job_processor.processed_df.columns:
                        company_counts = st.session_state.job_processor.processed_df['company'].value_counts().head(10)
                        
                        # Company distribution chart
                        fig = px.bar(
                            x=company_counts.values,
                            y=company_counts.index,
                            orientation='h',
                            title="Top 10 Companies by Job Listings",
                            labels={'x': 'Number of Job Listings', 'y': 'Company'}
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Company stats table
                        st.write("### Company Statistics")
                        company_data = []
                        for company, count in company_counts.head(10).items():
                            company_data.append({
                                'Company': company,
                                'Job Listings': count,
                                'Percentage': f"{(count/len(st.session_state.job_processor.processed_df)*100):.1f}%"
                            })
                        st.dataframe(pd.DataFrame(company_data), use_container_width=True)
                    else:
                        st.info("No company column found in the dataset")

                with tab4:
                    st.subheader("Location Analysis")
                    
                    if 'city_job_location' in st.session_state.job_processor.processed_df.columns:
                        location_counts = st.session_state.job_processor.processed_df['city_job_location'].value_counts().head(10)
                        
                        # Location distribution chart
                        fig = px.bar(
                            x=location_counts.values,
                            y=location_counts.index,
                            orientation='h',
                            title="Top 10 Cities by Job Listings",
                            labels={'x': 'Number of Job Listings', 'y': 'City'}
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Location stats table
                        st.write("### Location Statistics")
                        location_data = []
                        for location, count in location_counts.head(10).items():
                            location_data.append({
                                'City': location,
                                'Job Listings': count,
                                'Percentage': f"{(count/len(st.session_state.job_processor.processed_df)*100):.1f}%"
                            })
                        st.dataframe(pd.DataFrame(location_data), use_container_width=True)
                    else:
                        st.info("No city location column found in the dataset")


# RAG-Enhanced Job Query Interface
            if (st.session_state.rag_ready and st.session_state.openai_processor is not None 
                and st.session_state.openai_processor.is_available()):

                st.header("🤖 Intelligent Job Market Queries with RAG")
                st.markdown("Ask sophisticated questions about the job market with **Retrieval-Augmented Generation**!")

                # RAG system status
                col1, col2, col3 = st.columns(3)
                with col1:
                    vector_stats = st.session_state.job_processor.vector_store.get_stats()
                    st.info(f"📚 Documents: {vector_stats['document_count']}")
                with col2:
                    st.info(f"🧠 Context: {'Active' if maintain_context else 'Disabled'}")
                with col3:
                    history_count = len(st.session_state.job_processor.conversation_manager.conversation_history)
                    st.info(f"💬 History: {history_count} exchanges")

                # Example questions for job data
                with st.expander("💡 Example Job Market Questions"):
                    st.markdown("""
                    **Company Analysis:**
                    - Which companies are hiring the most and in what locations?
                    - What types of roles are the top companies posting?
                    - Compare hiring patterns between different companies
                    
                    **Location & Market Insights:**
                    - What are the best cities for job opportunities in this dataset?
                    - Which locations offer the highest paying positions?
                    - Analyze the geographic distribution of different job types
                    
                    **Salary & Compensation:**
                    - What's the salary range for different types of positions?
                    - Which companies or locations offer the best compensation?
                    - Analyze salary trends across different job categories
                    
                    **Career Guidance:**
                    - What skills or qualifications are most in demand?
                    - What career paths show the most opportunities?
                    - Based on this data, what advice would you give to job seekers?
                    
                    **Trend Analysis:**
                    - What patterns do you see in the job market data?
                    - Which industries or job functions are most represented?
                    - What insights can help with career planning?
                    """)

                # Query input
                question = st.text_area(
                    "Ask about the job market (RAG-enhanced analysis):",
                    placeholder="e.g., Which companies are hiring the most software engineers and what are the salary ranges?",
                    height=100
                )

                # Query options
                col1, col2 = st.columns(2)
                with col1:
                    show_retrieved_docs = st.checkbox("Show Retrieved Context", value=True,
                                                    help="Display job data retrieved by RAG")
                with col2:
                    show_conversation_history = st.checkbox("Show Conversation Context", value=False,
                                                          help="Display conversation history")

                if st.button("🚀 Analyze with RAG", type="primary", use_container_width=True) and question:
                    with st.spinner("Performing intelligent job market analysis..."):
                        # Process query with RAG
                        response = st.session_state.openai_processor.query_job_data_with_rag(
                            question,
                            st.session_state.job_processor.data_summary,
                            st.session_state.job_processor.vector_store,
                            st.session_state.job_processor.conversation_manager
                        )

                        # Display results
                        st.subheader("📊 Job Market Analysis Results")
                        st.write(response)

                        # Show retrieved context if requested
                        if show_retrieved_docs:
                            with st.expander("📄 Retrieved Job Data Context"):
                                retrieved_docs = st.session_state.job_processor.vector_store.search(question, k=rag_k_results)
                                if retrieved_docs:
                                    for i, doc in enumerate(retrieved_docs, 1):
                                        similarity = doc['score']
                                        text = doc['text']
                                        metadata = doc.get('metadata', {})
                                        
                                        st.write(f"**Document {i}** (Similarity: {similarity:.3f})")
                                        st.write(text)
                                        
                                        if metadata:
                                            if metadata.get('company'):
                                                st.caption(f"Company: {metadata['company']}")
                                            if metadata.get('location'):
                                                st.caption(f"Location: {metadata['location']}")
                                            if metadata.get('job_count'):
                                                st.caption(f"Job Count: {metadata['job_count']}")
                                        
                                        st.markdown("---")
                                else:
                                    st.write("No relevant documents retrieved")

                        # Show conversation context if requested
                        if show_conversation_history and maintain_context:
                            with st.expander("💬 Conversation History"):
                                history = st.session_state.job_processor.conversation_manager.conversation_history
                                if history:
                                    for i, exchange in enumerate(reversed(history[-5:]), 1):
                                        st.write(f"**Exchange {len(history) - i + 1}** ({exchange.get('query_type', 'general')})")
                                        st.write(f"*Q:* {exchange['question'][:150]}...")
                                        st.write(f"*A:* {exchange['answer'][:200]}...")
                                        st.caption(f"Time: {exchange['timestamp']}")
                                        st.markdown("---")
                                else:
                                    st.write("No conversation history yet")

                # RAG System Testing
                st.header("🔬 RAG System Testing & Exploration")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Vector Search Test")
                    test_query = st.text_input("Test job data search:", 
                                             placeholder="e.g., software engineer positions")
                    
                    if test_query and st.button("Search Job Documents"):
                        results = st.session_state.job_processor.vector_store.search(test_query, k=5)
                        if results:
                            for i, result in enumerate(results, 1):
                                st.write(f"**{i}.** Score: {result['score']:.3f}")
                                st.write(result['text'][:200] + "...")
                                if result.get('metadata', {}).get('company'):
                                    st.caption(f"Company: {result['metadata']['company']}")
                                st.markdown("---")
                        else:
                            st.write("No results found")

                with col2:
                    st.subheader("Query History")
                    history = st.session_state.job_processor.conversation_manager.conversation_history
                    if history:
                        for i, exchange in enumerate(reversed(history[-5:]), 1):
                            query_type = exchange.get('query_type', 'general')
                            with st.expander(f"{query_type.title()}: {exchange['question'][:40]}..."):
                                st.write(f"**Question:** {exchange['question']}")
                                st.write(f"**Answer:** {exchange['answer'][:300]}...")
                                st.caption(f"Type: {query_type} | Time: {exchange['timestamp']}")
                    else:
                        st.write("No query history yet")

                # Clear conversation history
                if st.button("🗑️ Clear Conversation History"):
                    st.session_state.job_processor.conversation_manager.clear_history()
                    st.success("Conversation history cleared")
                    st.rerun()

            elif st.session_state.job_processor.tokenized_df is not None:
                st.header("⚠️ RAG System Setup Required")
                if not st.session_state.openai_processor:
                    st.error("Please enter your OpenAI API key in the sidebar")
                elif not st.session_state.openai_processor.is_available():
                    st.error(f"OpenAI client error: {st.session_state.openai_processor._initialization_error}")
                else:
                    st.warning("Please build the RAG index to enable intelligent job market queries")

            # Export functionality
            if st.session_state.job_processor.tokenized_df is not None:
                st.header("📤 Export Enhanced Job Data")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    if st.button("📊 Download Processed Data"):
                        csv = st.session_state.job_processor.processed_df.to_csv(index=False)
                        st.download_button(
                            label="Download Processed CSV",
                            data=csv,
                            file_name="processed_job_data.csv",
                            mime="text/csv"
                        )

                with col2:
                    if st.button("🔤 Download Tokenized Data"):
                        csv = st.session_state.job_processor.tokenized_df.to_csv(index=False)
                        st.download_button(
                            label="Download Tokenized CSV",
                            data=csv,
                            file_name="tokenized_job_data.csv",
                            mime="text/csv"
                        )

                with col3:
                    if st.button("📚 Download RAG Documents"):
                        if st.session_state.job_processor.vector_store.documents:
                            rag_data = {
                                "documents": st.session_state.job_processor.vector_store.documents,
                                "metadata": st.session_state.job_processor.vector_store.metadata,
                                "total_documents": len(st.session_state.job_processor.vector_store.documents),
                                "vector_dimension": st.session_state.job_processor.vector_store.dimension
                            }
                            json_data = json.dumps(rag_data, indent=2)
                            st.download_button(
                                label="Download RAG Documents JSON",
                                data=json_data,
                                file_name="job_rag_documents.json",
                                mime="application/json"
                            )

                with col4:
                    if st.button("📈 Download Analysis Summary"):
                        if st.session_state.job_processor.data_summary:
                            summary_json = json.dumps(st.session_state.job_processor.data_summary, indent=2)
                            st.download_button(
                                label="Download Analysis JSON",
                                data=summary_json,
                                file_name="job_analysis_summary.json",
                                mime="application/json"
                            )

    # Footer
    st.markdown("---")
    st.markdown("""
    ## Job Data RAG Analyzer v3.0 - Complete Solution

    **Key Features:**
    - **Job-Specific Processing**: Optimized tokenization for company names, job titles, locations, and salaries
    - **Enhanced RAG System**: Vector search with job market context and conversation history
    - **Intelligent Analysis**: OpenAI-powered insights with retrieval-augmented generation
    - **Professional Interface**: Clean, intuitive design optimized for HR professionals and job seekers

    **Dependencies for Full Functionality:**
    ```bash
    pip install streamlit pandas numpy openai plotly nltk tiktoken usearch sentence-transformers
    ```
    
    **How to Run:**
    ```bash
    # Save all parts as job_rag_analyzer_v3.py, then run:
    streamlit run job_rag_analyzer_v3.py
    ```
    
    **Perfect for:**
    - HR professionals analyzing job market trends
    - Recruiters understanding hiring patterns
    - Job seekers researching opportunities
    - Data analysts exploring employment data
    - Career counselors providing guidance
    """)

if __name__ == "__main__":
    main()


