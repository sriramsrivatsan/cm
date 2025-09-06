import streamlit as st
import pandas as pd
import numpy as np
import openai
from io import StringIO
import json
import re
import os
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
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
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

def ensure_dir_for_file(path: str):
    """Ensure the directory for a given file path exists."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

# Set page config
st.set_page_config(
    page_title="Creative Professionals Job Data RAG Analyzer v3.1",
    page_icon="🎨",
    layout="wide"
)

# Enhanced creative software and tools mapping
CREATIVE_SOFTWARE_MAPPING = {
    # Adobe Creative Suite
    "adobe_suite": {
        "photoshop": ["Adobe Photoshop", "Photoshop", "PS", "PSD"],
        "lightroom": ["Adobe Lightroom", "Lightroom", "LR"],
        "illustrator": ["Adobe Illustrator", "Illustrator", "AI"],
        "indesign": ["Adobe InDesign", "InDesign", "ID"],
        "premiere": ["Adobe Premiere Pro", "Premiere Pro", "Premiere", "PPRO", "PR"],
        "after_effects": ["Adobe After Effects", "After Effects", "AE"],
        "xd": ["Adobe XD", "XD"],
        "dimension": ["Adobe Dimension"],
        "animate": ["Adobe Animate"],
        "audition": ["Adobe Audition"],
        "bridge": ["Adobe Bridge"],
        "character_animator": ["Adobe Character Animator"],
        "dreamweaver": ["Adobe Dreamweaver"],
        "fresco": ["Adobe Fresco"],
        "media_encoder": ["Adobe Media Encoder"],
        "prelude": ["Adobe Prelude"],
        "rush": ["Adobe Premiere Rush", "Premiere Rush"],
        "spark": ["Adobe Spark"],
        "substance": ["Adobe Substance", "Substance Painter", "Substance Designer"]
    },
    
    # Non-Adobe Creative Software
    "non_adobe": {
        "final_cut_pro": ["Final Cut Pro", "FCP", "FCPX"],
        "davinci_resolve": ["DaVinci Resolve", "Resolve", "Davinci"],
        "avid": ["Avid Media Composer", "Media Composer", "Avid"],
        "pro_tools": ["Pro Tools", "Protools"],
        "logic_pro": ["Logic Pro", "Logic"],
        "sketch": ["Sketch"],
        "figma": ["Figma"],
        "canva": ["Canva"],
        "gimp": ["GIMP"],
        "blender": ["Blender"],
        "maya": ["Autodesk Maya", "Maya"],
        "3ds_max": ["3ds Max", "3D Studio Max"],
        "cinema_4d": ["Cinema 4D", "C4D"],
        "zbrush": ["ZBrush"],
        "unity": ["Unity"],
        "unreal": ["Unreal Engine", "UE4", "UE5"],
        "procreate": ["Procreate"],
        "affinity": ["Affinity Designer", "Affinity Photo", "Affinity Publisher"],
        "capture_one": ["Capture One"],
        "luminar": ["Luminar"],
        "corel": ["CorelDRAW", "Corel Painter"],
        "keynote": ["Keynote"],
        "powerpoint": ["PowerPoint", "PPT"],
        "prezi": ["Prezi"],
        "invision": ["InVision"],
        "marvel": ["Marvel"],
        "principle": ["Principle"],
        "framer": ["Framer"],
        "webflow": ["Webflow"],
        "wordpress": ["WordPress"],
        "squarespace": ["Squarespace"],
        "wix": ["Wix"]
    },
    
    # AI/Creative AI Tools
    "ai_tools": {
        "midjourney": ["Midjourney"],
        "dall_e": ["DALL-E", "DALLE", "Dall-E"],
        "stable_diffusion": ["Stable Diffusion"],
        "chatgpt": ["ChatGPT", "GPT-4", "GPT-3"],
        "runway": ["Runway ML", "Runway"],
        "adobe_sensei": ["Adobe Sensei"],
        "topaz": ["Topaz Labs"],
        "gigapixel": ["Gigapixel AI"],
        "descript": ["Descript"],
        "synthesia": ["Synthesia"],
        "lumen5": ["Lumen5"],
        "jasper": ["Jasper AI"],
        "copy_ai": ["Copy.ai"],
        "notion_ai": ["Notion AI"]
    }
}

# Job role categorization patterns
JOB_ROLE_PATTERNS = {
    "designer": {
        "patterns": [
            r'\b(graphic|visual|ui|ux|web|digital|brand|creative|product|motion)\s*(designer?)\b',
            r'\bdesigner?\b',
            r'\b(art\s*director|creative\s*director)\b',
            r'\b(brand|visual|graphic)\s*(specialist|artist)\b'
        ],
        "keywords": ["design", "designer", "graphic", "visual", "ui", "ux", "creative", "art director"]
    },
    
    "video_professional": {
        "patterns": [
            r'\b(video|motion)\s*(editor?|producer?|artist)\b',
            r'\b(film|movie|cinema)\s*(editor?|producer?|maker)\b',
            r'\b(animator|animation)\b',
            r'\b(videographer|cinematographer)\b',
            r'\b(post[\s-]*production|vfx|visual\s*effects)\b'
        ],
        "keywords": ["video", "motion", "film", "animation", "editor", "producer", "vfx", "cinematographer"]
    },
    
    "photo_professional": {
        "patterns": [
            r'\b(photo|photography)\s*(editor?|retoucher?|specialist)\b',
            r'\bphotographer\b',
            r'\b(photo|image)\s*(manipulation|enhancement|processing)\b',
            r'\bretoucher?\b'
        ],
        "keywords": ["photo", "photography", "photographer", "retoucher", "image", "photo editing"]
    },
    
    "creative_professional": {
        "patterns": [
            r'\b(creative|content)\s*(professional|specialist|manager|director)\b',
            r'\b(multimedia|digital\s*media|content)\s*(artist|creator|specialist)\b',
            r'\b(marketing|brand)\s*(creative|designer|specialist)\b'
        ],
        "keywords": ["creative", "content", "multimedia", "digital media", "marketing creative"]
    }
}

# Creative skills and soft skills mapping
CREATIVE_SKILLS = {
    "technical_skills": [
        "color theory", "typography", "layout", "composition", "branding", "logo design",
        "web design", "mobile design", "responsive design", "user experience", "user interface",
        "wireframing", "prototyping", "mockups", "storyboarding", "concept development",
        "photo retouching", "image manipulation", "color correction", "photo editing",
        "video editing", "motion graphics", "animation", "visual effects", "compositing",
        "3d modeling", "rendering", "texturing", "lighting", "rigging"
    ],
    
    "soft_skills": [
        "collaboration", "communication", "presentation", "project management", "time management",
        "creativity", "problem solving", "attention to detail", "adaptability", "teamwork",
        "client management", "feedback incorporation", "deadline management", "multitasking",
        "critical thinking", "artistic vision", "trend awareness", "brand understanding"
    ],
    
    "creative_tasks": [
        "brand identity", "logo creation", "marketing materials", "social media graphics",
        "website design", "app design", "print design", "packaging design", "illustration",
        "photo shoot", "product photography", "portrait photography", "event photography",
        "video production", "commercial videos", "explainer videos", "social media videos",
        "documentary", "promotional content", "advertising campaigns", "content creation"
    ]
}

def detect_creative_software_and_skills(text: str) -> Dict[str, List[str]]:
    """Enhanced detection of creative software, AI tools, and skills in job descriptions"""
    if not text or pd.isna(text):
        return {"adobe_apps": [], "non_adobe_apps": [], "ai_tools": [], "technical_skills": [], "soft_skills": [], "creative_tasks": []}
    
    text_lower = text.lower()
    results = {
        "adobe_apps": [],
        "non_adobe_apps": [],
        "ai_tools": [],
        "technical_skills": [],
        "soft_skills": [],
        "creative_tasks": []
    }
    
    # Detect Adobe apps
    for app_key, variations in CREATIVE_SOFTWARE_MAPPING["adobe_suite"].items():
        for variation in variations:
            if re.search(r'\b' + re.escape(variation.lower()) + r'\b', text_lower):
                results["adobe_apps"].append(variation)
                break
    
    # Detect non-Adobe apps
    for app_key, variations in CREATIVE_SOFTWARE_MAPPING["non_adobe"].items():
        for variation in variations:
            if re.search(r'\b' + re.escape(variation.lower()) + r'\b', text_lower):
                results["non_adobe_apps"].append(variation)
                break
    
    # Detect AI tools
    for ai_key, variations in CREATIVE_SOFTWARE_MAPPING["ai_tools"].items():
        for variation in variations:
            if re.search(r'\b' + re.escape(variation.lower()) + r'\b', text_lower):
                results["ai_tools"].append(variation)
                break
    
    # Detect technical skills
    for skill in CREATIVE_SKILLS["technical_skills"]:
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
            results["technical_skills"].append(skill)
    
    # Detect soft skills
    for skill in CREATIVE_SKILLS["soft_skills"]:
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
            results["soft_skills"].append(skill)
    
    # Detect creative tasks
    for task in CREATIVE_SKILLS["creative_tasks"]:
        if re.search(r'\b' + re.escape(task.lower()) + r'\b', text_lower):
            results["creative_tasks"].append(task)
    
    # Remove duplicates
    for key in results:
        results[key] = list(set(results[key]))
    
    return results

def categorize_job_role(job_title: str, job_description: str = "") -> Dict[str, bool]:
    """Categorize job roles based on title and description"""
    combined_text = f"{job_title} {job_description}".lower()
    
    categories = {
        "is_designer": False,
        "is_video_professional": False,
        "is_photo_professional": False,
        "is_creative_professional": False
    }
    
    for role_type, config in JOB_ROLE_PATTERNS.items():
        # Check patterns
        for pattern in config["patterns"]:
            if re.search(pattern, combined_text, re.IGNORECASE):
                categories[f"is_{role_type}"] = True
                break
        
        # Check keywords if pattern didn't match
        if not categories[f"is_{role_type}"]:
            for keyword in config["keywords"]:
                if keyword.lower() in combined_text:
                    categories[f"is_{role_type}"] = True
                    break
    
    return categories


class RAGVectorStore:
    """Enhanced vector store for creative professionals job analysis"""
    
    def __init__(self):
        self.embedder = None
        self.index = None
        self.documents = []
        self.metadata = []
        self.dimension = 384
        self._index_built = False
        self._initialization_error = None
        self.backend_preference = 'usearch'
        self.backend = None
        
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
            
        # Initialize index
        try:
            self._init_index()
            st.success("Vector index initialized successfully")
        except Exception as e:
            self._initialization_error = f"Index error: {str(e)}"
            st.error(f"Error initializing vector index: {str(e)}")

    def _init_index(self):
        """Initialize the vector index according to backend preference"""
        self.index = None
        self.backend = None
        try:
            if getattr(self, 'backend_preference', 'usearch') == 'usearch' and USEARCH_AVAILABLE:
                self.index = Index(ndim=self.dimension, metric='cos', dtype=np.float32)
                self.backend = 'usearch'
            elif getattr(self, 'backend_preference', 'usearch') == 'faiss' and FAISS_AVAILABLE:
                import faiss
                self.index = faiss.IndexFlatIP(self.dimension)
                self.backend = 'faiss'
            else:
                if USEARCH_AVAILABLE:
                    self.index = Index(ndim=self.dimension, metric='cos', dtype=np.float32)
                    self.backend = 'usearch'
                elif FAISS_AVAILABLE:
                    import faiss
                    self.index = faiss.IndexFlatIP(self.dimension)
                    self.backend = 'faiss'
                else:
                    self.index = None
                    self.backend = 'bruteforce'
        except Exception as e:
            logger.error(f"Index initialization error: {str(e)}")
            self.index = None
            self.backend = 'bruteforce'
    
    def is_available(self) -> bool:
        """Check if RAG functionality is available"""
        return (self.embedder is not None and 
                self.index is not None and 
                self._initialization_error is None)
    
    def build_index(self, chunks: List[Dict]) -> bool:
        """Build vector index from creative job data chunks"""
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
            
            st.info(f"Building vector index for {len(texts)} creative job documents...")
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
                            
                            if getattr(self, 'backend', 'usearch') == 'usearch':
                                self.index.add(doc_id, embedding_vector)
                                self.documents.append(batch_texts[j])
                                self.metadata.append(batch_chunks[j].get('metadata', {}))
                            elif getattr(self, 'backend', '') == 'faiss':
                                # faiss expects 2D array for add
                                try:
                                    self.index.add(np.expand_dims(embedding_vector, axis=0))
                                    self.documents.append(batch_texts[j])
                                    self.metadata.append(batch_chunks[j].get('metadata', {}))
                                except Exception as e:
                                    logger.error(f'FAISS add error: {e}')
                            else:
                                # brute-force store embeddings
                                if not hasattr(self, '_brute_vectors'):
                                    self._brute_vectors = []
                                self._brute_vectors.append(embedding_vector.tolist())
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
            
            results = []
            
            # Perform search based on backend
            if getattr(self, 'backend', 'usearch') == 'faiss':
                D, I = self.index.search(np.array([query_embedding], dtype=np.float32), k)
                doc_ids = I[0]
                distances = D[0]
            elif getattr(self, 'backend', 'usearch') == 'usearch':
                search_results = self.index.search(query_embedding, k)
                # Handle different USearch result formats
                if hasattr(search_results, 'keys') and hasattr(search_results, 'distances'):
                    doc_ids = search_results.keys
                    distances = search_results.distances
                elif isinstance(search_results, tuple) and len(search_results) == 2:
                    doc_ids, distances = search_results
                else:
                    doc_ids = getattr(search_results, 'keys', [])
                    distances = getattr(search_results, 'distances', [])
            else:
                # brute-force search
                vecs = np.array(getattr(self, '_brute_vectors', []), dtype=np.float32)
                if vecs.size == 0:
                    return []
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                vecs = vecs / norms
                qv = query_embedding / max(1e-12, np.linalg.norm(query_embedding))
                sims = (vecs @ qv).tolist()
                idxs_scores = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:k]
                doc_ids = [i for i, s in idxs_scores]
                distances = [1.0 - s for i, s in idxs_scores]  # Convert similarity to distance
            
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
        """Create optimized document chunks from creative job data"""
        if not self.is_available():
            return []
            
        chunks = []
        
        try:
            # 1. Create enhanced job-level documents with creative analysis
            for idx, row in processed_df.iterrows():
                # Basic job info
                job_text_parts = []
                
                if pd.notna(row.get('company')):
                    job_text_parts.append(f"Company: {row['company']}")
                
                title = row.get('summary_job_title') or row.get('displayed_job_title') or row.get('Summary job title') or row.get('Displayed job title')
                if pd.notna(title):
                    job_text_parts.append(f"Job Title: {title}")
                
                # Location information
                location_parts = []
                for loc_col in ['city_job_location', 'City job location']:
                    if loc_col in row and pd.notna(row[loc_col]):
                        location_parts.append(row[loc_col])
                        break
                
                for loc_col in ['state_job_location', 'State job location']:
                    if loc_col in row and pd.notna(row[loc_col]):
                        location_parts.append(row[loc_col])
                        break
                
                for loc_col in ['country_job_location', 'Country job location']:
                    if loc_col in row and pd.notna(row[loc_col]):
                        location_parts.append(row[loc_col])
                        break
                
                if location_parts:
                    job_text_parts.append(f"Location: {', '.join(location_parts)}")
                
                # Job description with creative analysis
                description = row.get('job_description') or row.get('Job Description')
                if pd.notna(description):
                    desc_text = str(description)[:1000]  # Increased limit for creative jobs
                    job_text_parts.append(f"Description: {desc_text}")
                    
                    # Analyze creative requirements
                    creative_analysis = detect_creative_software_and_skills(desc_text)
                    
                    if creative_analysis['adobe_apps']:
                        job_text_parts.append(f"Adobe Apps Required: {', '.join(creative_analysis['adobe_apps'])}")
                    
                    if creative_analysis['non_adobe_apps']:
                        job_text_parts.append(f"Non-Adobe Apps Required: {', '.join(creative_analysis['non_adobe_apps'])}")
                    
                    if creative_analysis['ai_tools']:
                        job_text_parts.append(f"AI Tools Required: {', '.join(creative_analysis['ai_tools'])}")
                    
                    if creative_analysis['technical_skills']:
                        job_text_parts.append(f"Technical Skills: {', '.join(creative_analysis['technical_skills'][:5])}")
                    
                    if creative_analysis['soft_skills']:
                        job_text_parts.append(f"Soft Skills: {', '.join(creative_analysis['soft_skills'][:5])}")
                
                # Salary information
                salary = row.get('job_salary') or row.get('Job salary')
                if pd.notna(salary):
                    job_text_parts.append(f"Salary: {salary}")
                
                # Date information
                date = row.get('date') or row.get('Date')
                if pd.notna(date):
                    job_text_parts.append(f"Date: {date}")
                
                job_text = ". ".join(job_text_parts)
                
                # Categorize job role
                job_categories = categorize_job_role(str(title), str(description) if pd.notna(description) else "")
                
                chunks.append({
                    'text': job_text,
                    'type': 'job_listing',
                    'job_id': idx,
                    'metadata': {
                        'company': str(row.get('company', 'Unknown')),
                        'title': str(title) if pd.notna(title) else 'Unknown',
                        'location': ', '.join(location_parts) if location_parts else 'Unknown',
                        'row_index': idx,
                        'is_designer': job_categories.get('is_designer', False),
                        'is_video_professional': job_categories.get('is_video_professional', False),
                        'is_photo_professional': job_categories.get('is_photo_professional', False),
                        'is_creative_professional': job_categories.get('is_creative_professional', False),
                        'adobe_apps': creative_analysis.get('adobe_apps', []),
                        'non_adobe_apps': creative_analysis.get('non_adobe_apps', []),
                        'ai_tools': creative_analysis.get('ai_tools', []),
                        'technical_skills': creative_analysis.get('technical_skills', []),
                        'soft_skills': creative_analysis.get('soft_skills', [])
                    }
                })
            
            # 2. Create aggregated chunks for specific analysis questions
            
            # Adobe vs Non-Adobe analysis chunks
            adobe_only_jobs = []
            non_adobe_only_jobs = []
            both_apps_jobs = []
            
            for idx, row in processed_df.iterrows():
                description = row.get('job_description') or row.get('Job Description')
                if pd.notna(description):
                    creative_analysis = detect_creative_software_and_skills(str(description))
                    has_adobe = len(creative_analysis.get('adobe_apps', [])) > 0
                    has_non_adobe = len(creative_analysis.get('non_adobe_apps', [])) > 0
                    
                    if has_adobe and not has_non_adobe:
                        adobe_only_jobs.append(idx)
                    elif has_non_adobe and not has_adobe:
                        non_adobe_only_jobs.append(idx)
                    elif has_adobe and has_non_adobe:
                        both_apps_jobs.append(idx)
            
            # Create summary chunks
            if non_adobe_only_jobs:
                non_adobe_text = f"Jobs requiring only non-Adobe applications: {len(non_adobe_only_jobs)} positions found"
                chunks.append({
                    'text': non_adobe_text,
                    'type': 'software_analysis',
                    'analysis_type': 'non_adobe_only',
                    'metadata': {'job_count': len(non_adobe_only_jobs), 'job_indices': non_adobe_only_jobs}
                })
            
            if both_apps_jobs:
                both_apps_text = f"Jobs requiring both Adobe and non-Adobe applications: {len(both_apps_jobs)} positions found"
                chunks.append({
                    'text': both_apps_text,
                    'type': 'software_analysis',
                    'analysis_type': 'both_adobe_and_non_adobe',
                    'metadata': {'job_count': len(both_apps_jobs), 'job_indices': both_apps_jobs}
                })
            
            # 3. Create role-specific aggregations
            designer_jobs = []
            video_jobs = []
            photo_jobs = []
            
            for idx, row in processed_df.iterrows():
                title = row.get('summary_job_title') or row.get('displayed_job_title') or row.get('Summary job title') or row.get('Displayed job title')
                description = row.get('job_description') or row.get('Job Description')
                
                job_categories = categorize_job_role(str(title) if pd.notna(title) else "", str(description) if pd.notna(description) else "")
                
                if job_categories.get('is_designer', False):
                    designer_jobs.append(idx)
                if job_categories.get('is_video_professional', False):
                    video_jobs.append(idx)
                if job_categories.get('is_photo_professional', False):
                    photo_jobs.append(idx)
            
            # Create role summary chunks
            if designer_jobs:
                designer_text = f"Designer roles identified: {len(designer_jobs)} positions including graphic designers, UI/UX designers, and creative professionals"
                chunks.append({
                    'text': designer_text,
                    'type': 'role_analysis',
                    'role_type': 'designer',
                    'metadata': {'job_count': len(designer_jobs), 'job_indices': designer_jobs}
                })
            
            if video_jobs:
                video_text = f"Video professional roles identified: {len(video_jobs)} positions including video editors, motion graphics artists, and videographers"
                chunks.append({
                    'text': video_text,
                    'type': 'role_analysis',
                    'role_type': 'video_professional',
                    'metadata': {'job_count': len(video_jobs), 'job_indices': video_jobs}
                })
            
            if photo_jobs:
                photo_text = f"Photo professional roles identified: {len(photo_jobs)} positions including photographers, photo editors, and retouchers"
                chunks.append({
                    'text': photo_text,
                    'type': 'role_analysis',
                    'role_type': 'photo_professional',
                    'metadata': {'job_count': len(photo_jobs), 'job_indices': photo_jobs}
                })
            
            # 4. Create AI tools analysis chunk
            ai_tool_jobs = []
            for idx, row in processed_df.iterrows():
                description = row.get('job_description') or row.get('Job Description')
                if pd.notna(description):
                    creative_analysis = detect_creative_software_and_skills(str(description))
                    if creative_analysis.get('ai_tools'):
                        ai_tool_jobs.append(idx)
            
            if ai_tool_jobs:
                ai_text = f"Jobs requiring AI tools: {len(ai_tool_jobs)} positions mention AI tools like ChatGPT, Midjourney, DALL-E, or other creative AI platforms"
                chunks.append({
                    'text': ai_text,
                    'type': 'ai_analysis',
                    'metadata': {'job_count': len(ai_tool_jobs), 'job_indices': ai_tool_jobs}
                })
            
            # 5. Create dataset overview
            total_jobs = len(processed_df)
            unique_companies = processed_df['company'].nunique() if 'company' in processed_df.columns else 0
            
            overview_text = f"Creative professionals job dataset contains {total_jobs} total job listings"
            if unique_companies > 0:
                overview_text += f" from {unique_companies} different companies"
            
            chunks.append({
                'text': overview_text,
                'type': 'dataset_overview',
                'metadata': {
                    'total_jobs': total_jobs,
                    'unique_companies': unique_companies,
                    'designer_jobs': len(designer_jobs),
                    'video_jobs': len(video_jobs),
                    'photo_jobs': len(photo_jobs),
                    'ai_tool_jobs': len(ai_tool_jobs)
                }
            })
            
            st.info(f"Created {len(chunks)} enhanced document chunks for creative job analysis")
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
            'initialization_error': self._initialization_error,
            'backend': self.backend
        }


class CreativeJobTokenizer:
    """Enhanced tokenizer specialized for creative professional job data"""
    
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
        """Enhanced text tokenization for creative job descriptions"""
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

    def tokenize_creative_job_description(self, description: str) -> Dict[str, List[str]]:
        """Enhanced tokenization specifically for creative job descriptions"""
        if pd.isna(description) or not isinstance(description, str):
            return {
                'basic_tokens': [],
                'adobe_apps': [],
                'non_adobe_apps': [],
                'ai_tools': [],
                'technical_skills': [],
                'soft_skills': [],
                'creative_tasks': [],
                'job_categories': []
            }

        # Get creative analysis
        creative_analysis = detect_creative_software_and_skills(description)
        
        # Basic tokenization
        basic_tokens = self.tokenize_text(description, method='lemmatize')
        
        # Job role categorization
        job_categories = categorize_job_role("", description)
        category_tokens = []
        
        for category, is_category in job_categories.items():
            if is_category:
                category_tokens.append(category.replace('is_', ''))
        
        return {
            'basic_tokens': basic_tokens,
            'adobe_apps': creative_analysis.get('adobe_apps', []),
            'non_adobe_apps': creative_analysis.get('non_adobe_apps', []),
            'ai_tools': creative_analysis.get('ai_tools', []),
            'technical_skills': creative_analysis.get('technical_skills', []),
            'soft_skills': creative_analysis.get('soft_skills', []),
            'creative_tasks': creative_analysis.get('creative_tasks', []),
            'job_categories': category_tokens
        }

    def tokenize_job_title(self, title: str) -> List[str]:
        """Enhanced tokenization for creative job titles"""
        if pd.isna(title) or not isinstance(title, str):
            return ['unknown_title']

        tokens = []
        
        try:
            # Basic tokenization
            basic_tokens = self.tokenize_text(title, method='lemmatize')
            tokens.extend(basic_tokens)
            
            # Add title-specific context
            tokens.append('job_title')
            
            # Check for creative role categories
            job_categories = categorize_job_role(title, "")
            for category, is_category in job_categories.items():
                if is_category:
                    tokens.append(category.replace('is_', '') + '_role')
            
            # Add seniority level indicators
            title_lower = title.lower()
            if any(word in title_lower for word in ['senior', 'sr', 'lead', 'principal']):
                tokens.extend(['senior_level', 'experienced'])
            elif any(word in title_lower for word in ['junior', 'jr', 'entry', 'associate']):
                tokens.extend(['junior_level', 'entry_level'])
            elif any(word in title_lower for word in ['manager', 'director', 'head', 'chief']):
                tokens.extend(['management', 'leadership'])
            
            # Add domain indicators for creative roles
            if any(word in title_lower for word in ['designer', 'design', 'creative']):
                tokens.extend(['creative_role', 'design_professional'])
            elif any(word in title_lower for word in ['video', 'motion', 'film', 'animation']):
                tokens.extend(['video_professional', 'motion_specialist'])
            elif any(word in title_lower for word in ['photo', 'photographer', 'retoucher']):
                tokens.extend(['photo_professional', 'imaging_specialist'])
                
        except Exception as e:
            logger.error(f"Error tokenizing job title {title}: {str(e)}")
            tokens = ['job_title', 'unknown']

        return tokens

    def tokenize_company(self, company: str) -> List[str]:
        """Tokenize company names with creative industry context"""
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
            
            # Industry indicators for creative companies
            company_lower = company.lower()
            if any(word in company_lower for word in ['design', 'creative', 'agency', 'studio']):
                tokens.extend(['creative_company', 'design_agency'])
            elif any(word in company_lower for word in ['media', 'production', 'film', 'video']):
                tokens.extend(['media_company', 'production_house'])
            elif any(word in company_lower for word in ['advertising', 'marketing', 'brand']):
                tokens.extend(['advertising_agency', 'marketing_company'])
            
        except Exception as e:
            logger.error(f"Error tokenizing company {company}: {str(e)}")
            tokens = ['company_name', 'unknown']

        return tokens


class CreativeJobConversationManager:
    """Enhanced conversation manager for creative job queries"""
    
    def __init__(self):
        self.conversation_history = []
        self.max_history_length = 20  # Increased for complex creative queries
        
    def add_exchange(self, question: str, answer: str, context_used: List[str] = None):
        """Add a question-answer exchange to conversation history"""
        exchange = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'context_used': context_used or [],
            'query_type': self._classify_creative_query(question)
        }
        
        self.conversation_history.append(exchange)
        
        # Trim history if too long
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def _classify_creative_query(self, question: str) -> str:
        """Classify the type of creative job-related query"""
        question_lower = question.lower()
        
        # Software-specific queries
        if any(word in question_lower for word in ['adobe', 'photoshop', 'illustrator', 'premiere', 'after effects']):
            return 'adobe_software_query'
        elif any(word in question_lower for word in ['non-adobe', 'figma', 'sketch', 'final cut', 'davinci']):
            return 'non_adobe_software_query'
        elif any(word in question_lower for word in ['ai tool', 'chatgpt', 'midjourney', 'dall']):
            return 'ai_tools_query'
        
        # Role-specific queries
        elif any(word in question_lower for word in ['designer', 'design', 'graphic', 'ui', 'ux']):
            return 'designer_query'
        elif any(word in question_lower for word in ['video', 'motion', 'film', 'animation']):
            return 'video_professional_query'
        elif any(word in question_lower for word in ['photo', 'photographer', 'retoucher']):
            return 'photo_professional_query'
        
        # Analysis-specific queries
        elif any(word in question_lower for word in ['how many', 'count', 'number']):
            return 'count_query'
        elif any(word in question_lower for word in ['top', 'most', 'popular', 'common']):
            return 'ranking_query'
        elif any(word in question_lower for word in ['skill', 'requirement', 'soft skill']):
            return 'skills_query'
        elif any(word in question_lower for word in ['industry', 'company', 'hiring']):
            return 'industry_query'
        elif any(word in question_lower for word in ['summary', 'overview', 'analyze', 'insight']):
            return 'analysis_query'
        else:
            return 'general_creative_query'
    
    def get_context_for_query(self, current_question: str) -> str:
        """Get relevant conversation context for creative job queries"""
        if not self.conversation_history:
            return ""
        
        current_type = self._classify_creative_query(current_question)
        
        # Get recent relevant exchanges
        relevant_history = []
        for exchange in reversed(self.conversation_history[-7:]):
            if (exchange['query_type'] == current_type or 
                exchange['query_type'] == 'analysis_query' or
                'software' in exchange['query_type'] and 'software' in current_type):
                relevant_history.append(exchange)
        
        if not relevant_history:
            # Fallback to recent history
            relevant_history = self.conversation_history[-3:]
        
        context_parts = ["Previous relevant creative job analysis:"]
        for i, exchange in enumerate(relevant_history[:3], 1):
            context_parts.append(f"Q{i}: {exchange['question'][:120]}...")
            context_parts.append(f"A{i}: {exchange['answer'][:180]}...")
        
        return "\n".join(context_parts)
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


def analyze_creative_job_dataset(df: pd.DataFrame) -> Dict:
    """Comprehensive analysis of creative job dataset for answering specific questions"""
    analysis_results = {
        'total_jobs': len(df),
        'adobe_analysis': {},
        'role_analysis': {},
        'software_analysis': {},
        'ai_tools_analysis': {},
        'skills_analysis': {},
        'industry_analysis': {}
    }
    
    try:
        # Initialize counters
        adobe_only_jobs = []
        non_adobe_only_jobs = []
        both_apps_jobs = []
        ai_tool_jobs = []
        
        # Role counters
        designer_jobs = []
        video_jobs = []
        photo_jobs = []
        
        # Skills counters
        all_technical_skills = []
        all_soft_skills = []
        all_creative_tasks = []
        
        # Software counters
        all_adobe_apps = []
        all_non_adobe_apps = []
        all_ai_tools = []
        
        # Process each job
        for idx, row in df.iterrows():
            # Get job description
            description = row.get('job_description') or row.get('Job Description', '')
            title = row.get('summary_job_title') or row.get('displayed_job_title') or row.get('Summary job title') or row.get('Displayed job title', '')
            company = row.get('company', '')
            
            if pd.notna(description):
                # Creative analysis
                creative_analysis = detect_creative_software_and_skills(str(description))
                
                # Software analysis
                has_adobe = len(creative_analysis.get('adobe_apps', [])) > 0
                has_non_adobe = len(creative_analysis.get('non_adobe_apps', [])) > 0
                has_ai_tools = len(creative_analysis.get('ai_tools', [])) > 0
                
                if has_adobe and not has_non_adobe:
                    adobe_only_jobs.append({
                        'index': idx,
                        'title': title,
                        'company': company,
                        'adobe_apps': creative_analysis.get('adobe_apps', [])
                    })
                elif has_non_adobe and not has_adobe:
                    non_adobe_only_jobs.append({
                        'index': idx,
                        'title': title,
                        'company': company,
                        'non_adobe_apps': creative_analysis.get('non_adobe_apps', [])
                    })
                elif has_adobe and has_non_adobe:
                    both_apps_jobs.append({
                        'index': idx,
                        'title': title,
                        'company': company,
                        'adobe_apps': creative_analysis.get('adobe_apps', []),
                        'non_adobe_apps': creative_analysis.get('non_adobe_apps', [])
                    })
                
                if has_ai_tools:
                    ai_tool_jobs.append({
                        'index': idx,
                        'title': title,
                        'company': company,
                        'ai_tools': creative_analysis.get('ai_tools', [])
                    })
                
                # Collect all software mentions
                all_adobe_apps.extend(creative_analysis.get('adobe_apps', []))
                all_non_adobe_apps.extend(creative_analysis.get('non_adobe_apps', []))
                all_ai_tools.extend(creative_analysis.get('ai_tools', []))
                
                # Collect skills
                all_technical_skills.extend(creative_analysis.get('technical_skills', []))
                all_soft_skills.extend(creative_analysis.get('soft_skills', []))
                all_creative_tasks.extend(creative_analysis.get('creative_tasks', []))
            
            # Role categorization
            job_categories = categorize_job_role(str(title), str(description))
            
            if job_categories.get('is_designer', False):
                designer_jobs.append({
                    'index': idx,
                    'title': title,
                    'company': company,
                    'description_snippet': str(description)[:200] if pd.notna(description) else ''
                })
            
            if job_categories.get('is_video_professional', False):
                video_jobs.append({
                    'index': idx,
                    'title': title,
                    'company': company,
                    'description_snippet': str(description)[:200] if pd.notna(description) else ''
                })
            
            if job_categories.get('is_photo_professional', False):
                photo_jobs.append({
                    'index': idx,
                    'title': title,
                    'company': company,
                    'description_snippet': str(description)[:200] if pd.notna(description) else ''
                })
        
        # Compile results
        analysis_results['adobe_analysis'] = {
            'adobe_only_count': len(adobe_only_jobs),
            'non_adobe_only_count': len(non_adobe_only_jobs),
            'both_apps_count': len(both_apps_jobs),
            'adobe_only_jobs': adobe_only_jobs,
            'non_adobe_only_jobs': non_adobe_only_jobs,
            'both_apps_jobs': both_apps_jobs
        }
        
        analysis_results['role_analysis'] = {
            'designer_count': len(designer_jobs),
            'video_professional_count': len(video_jobs),
            'photo_professional_count': len(photo_jobs),
            'designer_jobs': designer_jobs,
            'video_jobs': video_jobs,
            'photo_jobs': photo_jobs
        }
        
        analysis_results['software_analysis'] = {
            'adobe_apps_frequency': Counter(all_adobe_apps),
            'non_adobe_apps_frequency': Counter(all_non_adobe_apps),
            'photoshop_count': Counter(all_adobe_apps).get('Photoshop', 0) + Counter(all_adobe_apps).get('Adobe Photoshop', 0)
        }
        
        analysis_results['ai_tools_analysis'] = {
            'ai_tools_count': len(ai_tool_jobs),
            'ai_tools_jobs': ai_tool_jobs,
            'ai_tools_frequency': Counter(all_ai_tools)
        }
        
        analysis_results['skills_analysis'] = {
            'technical_skills_frequency': Counter(all_technical_skills),
            'soft_skills_frequency': Counter(all_soft_skills),
            'creative_tasks_frequency': Counter(all_creative_tasks)
        }
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Error in creative job analysis: {str(e)}")
        return analysis_results


class CreativeJobOpenAIProcessor:
    """Enhanced OpenAI processor optimized for creative job analysis"""
    
    def __init__(self, api_key: str):
        self._api_key = api_key
        self.client = None
        self.tokenizer = None
        self.max_context_length = 16385
        self.max_completion_tokens = 3000  # Increased for detailed creative analysis
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

    def generate_creative_job_system_prompt(self, data_summary: Dict) -> str:
        """Generate system prompt optimized for creative job analysis"""
        try:
            dataset_info = data_summary.get('dataset_info', {})
            prompt = f"""You are an expert creative industry analyst specializing in job market analysis for creative professionals (CPros), with deep knowledge of:

Dataset Overview: {dataset_info.get('total_rows', 0):,} creative job listings across {dataset_info.get('total_columns', 0)} data fields
Key Focus Areas: Adobe vs Non-Adobe software requirements, AI tools adoption, creative roles analysis

Your specialized expertise includes:
- Creative software analysis (Adobe Creative Suite vs alternatives like Figma, Sketch, Final Cut Pro, DaVinci Resolve)
- AI tools in creative workflows (ChatGPT, Midjourney, DALL-E, Runway ML, etc.)
- Creative role categorization (designers, video professionals, photo professionals)
- Industry hiring patterns and skill requirements
- Creative software market trends and professional development

You have access to a RAG system providing relevant creative job data context. Use this context to provide:

1. **Quantitative Analysis**: Provide exact counts, percentages, and statistical breakdowns
2. **Software Trends**: Compare Adobe vs non-Adobe usage patterns
3. **Role-Specific Insights**: Analyze requirements by creative discipline
4. **Industry Intelligence**: Identify hiring companies and emerging trends
5. **Professional Guidance**: Offer career advice based on market data

Instructions for Creative Job Analysis:
- Always cite specific job counts and percentages from the dataset
- Compare Adobe Creative Suite vs alternative software adoption
- Identify cross-disciplinary skill requirements
- Highlight AI tool integration in creative workflows
- Provide actionable insights for creative professionals
- Present data in clear, professional tables when appropriate
- Focus on practical implications for career development

When answering questions about creative roles and software requirements, structure responses with:
- Executive Summary (key findings)
- Detailed Analysis (with specific counts)
- Software Breakdown (Adobe vs Non-Adobe vs AI tools)
- Professional Recommendations"""

            # Ensure system prompt fits within limits
            max_system_tokens = int(self.max_input_tokens * 0.4)
            return self.truncate_text(prompt, max_system_tokens)
            
        except Exception as e:
            logger.error(f"Error generating system prompt: {str(e)}")
            return "You are a creative industry job market analyst. Help analyze creative professional job datasets and provide insights."

    def prepare_creative_rag_context(self, retrieved_docs: List[Dict], conversation_context: str = "", dataset_analysis: Dict = None) -> str:
        """Prepare RAG context optimized for creative job analysis"""
        context_parts = []
        
        try:
            if conversation_context:
                context_parts.append(f"Previous Conversation:\n{conversation_context}")
            
            # Add dataset analysis summary if available
            if dataset_analysis:
                context_parts.append("Dataset Analysis Summary:")
                
                # Role counts
                role_analysis = dataset_analysis.get('role_analysis', {})
                if role_analysis:
                    context_parts.append(f"Designer roles: {role_analysis.get('designer_count', 0)}")
                    context_parts.append(f"Video professionals: {role_analysis.get('video_professional_count', 0)}")
                    context_parts.append(f"Photo professionals: {role_analysis.get('photo_professional_count', 0)}")
                
                # Software analysis
                adobe_analysis = dataset_analysis.get('adobe_analysis', {})
                if adobe_analysis:
                    context_parts.append(f"Adobe-only jobs: {adobe_analysis.get('adobe_only_count', 0)}")
                    context_parts.append(f"Non-Adobe only jobs: {adobe_analysis.get('non_adobe_only_count', 0)}")
                    context_parts.append(f"Both Adobe and non-Adobe: {adobe_analysis.get('both_apps_count', 0)}")
                
                # AI tools
                ai_analysis = dataset_analysis.get('ai_tools_analysis', {})
                if ai_analysis:
                    context_parts.append(f"Jobs requiring AI tools: {ai_analysis.get('ai_tools_count', 0)}")
            
            if retrieved_docs:
                context_parts.append("Relevant Creative Job Data:")
                
                # Group documents by type for better organization
                job_listings = [doc for doc in retrieved_docs if doc.get('metadata', {}).get('company')]
                analysis_docs = [doc for doc in retrieved_docs if 'analysis' in doc.get('text', '')]
                other_docs = [doc for doc in retrieved_docs if doc not in job_listings + analysis_docs]
                
                # Add specific job listings
                if job_listings:
                    context_parts.append("\nSpecific Creative Job Listings:")
                    for i, doc in enumerate(job_listings[:4], 1):
                        score = doc.get('score', 0)
                        metadata = doc.get('metadata', {})
                        context_parts.append(f"{i}. [Relevance: {score:.3f}] {doc.get('text', '')}")
                        
                        # Add creative-specific metadata
                        if metadata.get('adobe_apps'):
                            context_parts.append(f"   Adobe Apps: {', '.join(metadata['adobe_apps'])}")
                        if metadata.get('non_adobe_apps'):
                            context_parts.append(f"   Non-Adobe Apps: {', '.join(metadata['non_adobe_apps'])}")
                        if metadata.get('ai_tools'):
                            context_parts.append(f"   AI Tools: {', '.join(metadata['ai_tools'])}")
                        if metadata.get('is_designer'):
                            context_parts.append("   Role Type: Designer")
                        elif metadata.get('is_video_professional'):
                            context_parts.append("   Role Type: Video Professional")
                        elif metadata.get('is_photo_professional'):
                            context_parts.append("   Role Type: Photo Professional")
                
                # Add analysis documents
                if analysis_docs:
                    context_parts.append("\nDataset Analysis:")
                    for doc in analysis_docs[:3]:
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

    def query_creative_data_with_rag(self, question: str, data_summary: Dict, vector_store: RAGVectorStore, 
                                   conversation_manager: CreativeJobConversationManager, dataset_analysis: Dict = None) -> str:
        """Process creative job queries with enhanced RAG"""
        
        if not self.is_available():
            return f"OpenAI client not available: {self._initialization_error}"
        
        try:
            # Get conversation context
            conversation_context = conversation_manager.get_context_for_query(question)
            
            # Retrieve relevant creative job documents
            retrieved_docs = []
            try:
                retrieved_docs = vector_store.search(question, k=10)  # More docs for creative analysis
                logger.info(f"Retrieved {len(retrieved_docs)} creative job documents for query")
            except Exception as search_error:
                logger.warning(f"Vector search failed: {search_error}")
            
            # Generate creative job system prompt
            system_prompt = self.generate_creative_job_system_prompt(data_summary)
            
            # Prepare enhanced RAG context with dataset analysis
            rag_context = self.prepare_creative_rag_context(retrieved_docs, conversation_context, dataset_analysis)
            
            # Create user message
            user_message = f"Creative Industry Query: {question}"
            if rag_context:
                user_message += f"\n\nRelevant Creative Job Data:\n{rag_context}"
            
            # Token management
            system_tokens = self.count_tokens(system_prompt)
            user_tokens = self.count_tokens(user_message)
            
            total_input_tokens = system_tokens + user_tokens
            
            if total_input_tokens > self.max_input_tokens:
                excess_tokens = total_input_tokens - self.max_input_tokens
                if rag_context:
                    current_context_tokens = self.count_tokens(rag_context)
                    reduced_context_tokens = max(500, current_context_tokens - excess_tokens)
                    rag_context = self.truncate_text(rag_context, reduced_context_tokens)
                    user_message = f"Creative Industry Query: {question}\n\nRelevant Creative Job Data:\n{rag_context}"
            
            # Calculate available completion tokens
            final_input_tokens = self.count_tokens(system_prompt) + self.count_tokens(user_message)
            available_tokens = self.max_context_length - final_input_tokens - 100
            completion_tokens = min(self.max_completion_tokens, max(1000, available_tokens))
            
            # Make API call
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=completion_tokens,
                    temperature=0.3,  # Lower temperature for more consistent analysis
                    timeout=45
                )
                
                answer = response.choices[0].message.content
                
                # Store in conversation history
                context_used = [doc['text'][:150] + "..." for doc in retrieved_docs[:3]]
                conversation_manager.add_exchange(question, answer, context_used)
                
                logger.info(f"Successfully processed creative job query: {question[:50]}...")
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
            logger.error(f"Error in creative job RAG query processing: {error_msg}")
            
            if "maximum context length" in error_msg.lower():
                return """The query is too complex for a single analysis. Please try:
1. Ask more specific questions about particular software or creative roles
2. Focus on specific aspects like Adobe vs non-Adobe requirements
3. Break down your analysis into smaller, focused questions

The creative job dataset has been successfully processed with enhanced RAG capabilities for creative professional analysis."""
            else:
                return f"Error processing creative job query: {error_msg}. Please try a simpler question or check your OpenAI API key."



class CreativeJobCSVProcessor:
    """Enhanced CSV processor specifically optimized for creative job listing data"""
    
    def __init__(self):
        self.df = None
        self.processed_df = None
        self.tokenized_df = None
        self.data_summary = None
        self.tokenization_summary = None
        self.dataset_analysis = None
        self.tokenizer = CreativeJobTokenizer()
        self.vector_store = RAGVectorStore()
        self.conversation_manager = CreativeJobConversationManager()

    def load_csv(self, uploaded_file) -> bool:
        """Load and validate creative job data CSV"""
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

            # Validate creative job data structure
            if self.df.empty:
                raise ValueError("CSV file is empty")
                
            if len(self.df.columns) == 0:
                raise ValueError("CSV file has no columns")

            # Check for expected creative job data columns
            expected_columns = ['company', 'job', 'title', 'description', 'location', 'salary']
            found_columns = []
            
            for col in self.df.columns:
                col_lower = col.lower().replace(' ', '_')
                for expected in expected_columns:
                    if expected in col_lower:
                        found_columns.append(expected)
                        break

            if len(found_columns) < 3:
                st.warning("Processing as general job dataset. Creative-specific analysis will still be performed.")
            else:
                st.success(f"Detected creative job dataset with columns: {', '.join(found_columns)}")

            logger.info(f"Loaded creative job CSV: {len(self.df)} rows, {len(self.df.columns)} columns")
            return True
            
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            logger.error(f"CSV loading error: {str(e)}")
            return False

    def clean_job_data(self) -> bool:
        """Clean and preprocess creative job data with domain-specific logic"""
        if self.df is None:
            st.error("No data loaded")
            return False

        try:
            self.processed_df = self.df.copy()

            # Standardize column names for creative job data
            column_mapping = {}
            used_names = set()
            
            for col in self.processed_df.columns:
                original_col = col
                clean_col = re.sub(r'[^\w\s]', '', str(col)).strip().replace(' ', '_').lower()
                
                # Map to standard creative job data column names
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

            # Handle missing values with creative job-specific logic
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

            # Clean text fields specifically
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
                        series_to_clean = series_to_clean.apply(lambda x: str(x).strip() if pd.notna(x) else 'Unknown')
                        series_to_clean = series_to_clean.apply(lambda x: re.sub(r'\s+', ' ', str(x)) if pd.notna(x) else 'Unknown')
                        
                        # Assign back to the DataFrame
                        self.processed_df[col] = series_to_clean
                        
                    except Exception as e:
                        logger.warning(f"Could not clean text column {col}: {e}")
                        try:
                            self.processed_df[col] = self.processed_df[col].astype(str)
                        except Exception as e2:
                            logger.error(f"Complete failure cleaning column {col}: {e2}")
                            continue

            # Handle date column if present
            if 'date' in self.processed_df.columns:
                try:
                    self.processed_df['date'] = pd.to_datetime(self.processed_df['date'], errors='coerce')
                    self.processed_df['date'] = self.processed_df['date'].fillna(pd.Timestamp('2023-01-01'))
                except Exception as e:
                    logger.warning(f"Could not convert date column: {e}")

            # Clean salary information
            if 'job_salary' in self.processed_df.columns:
                try:
                    self.processed_df['job_salary'] = self.processed_df['job_salary'].apply(
                        lambda x: re.sub(r'[\$,]', '', str(x)) if pd.notna(x) else 'Not Specified'
                    )
                except Exception as e:
                    logger.warning(f"Could not clean salary column: {e}")
                    self.processed_df['job_salary'] = self.processed_df['job_salary'].astype(str)

            logger.info(f"Creative job data cleaning completed: {len(self.processed_df)} rows, {len(self.processed_df.columns)} columns")
            return True
            
        except Exception as e:
            st.error(f"Error cleaning creative job data: {str(e)}")
            logger.error(f"Creative job data cleaning error: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

    def tokenize_creative_dataset(self) -> bool:
        """Perform creative job-specific tokenization of the dataset"""
        if self.processed_df is None:
            st.error("No processed data available")
            return False

        try:
            st.info("Starting creative job-specific tokenization process...")

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
                    # Creative job-specific tokenization based on column type
                    if col == 'company':
                        for value in self.processed_df[col]:
                            tokens = self.tokenizer.tokenize_company(value)
                            column_tokens.extend(tokens)

                    elif col in ['summary_job_title', 'displayed_job_title']:
                        for value in self.processed_df[col]:
                            tokens = self.tokenizer.tokenize_job_title(value)
                            column_tokens.extend(tokens)

                    elif col == 'job_description':
                        # Enhanced creative job description tokenization
                        for value in self.processed_df[col]:
                            creative_tokens = self.tokenizer.tokenize_creative_job_description(value)
                            # Flatten all creative analysis results
                            for token_type, tokens in creative_tokens.items():
                                column_tokens.extend(tokens)

                    elif col in ['city_job_location', 'state_job_location', 'country_job_location']:
                        location_type = col.split('_')[0]
                        for value in self.processed_df[col]:
                            tokens = self.tokenizer.tokenize_text(value, method='lemmatize')
                            tokens.extend([f'{location_type}_location', 'geographic_data'])
                            column_tokens.extend(tokens)

                    elif col == 'job_salary':
                        for value in self.processed_df[col]:
                            tokens = self.tokenizer.tokenize_text(str(value))
                            tokens.extend(['salary_info', 'compensation'])
                            column_tokens.extend(tokens)

                    elif pd.api.types.is_datetime64_any_dtype(self.processed_df[col]):
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
                    
                    if col == 'job_description':
                        for value in self.processed_df[col]:
                            creative_tokens = self.tokenizer.tokenize_creative_job_description(value)
                            all_desc_tokens = []
                            for token_type, tokens in creative_tokens.items():
                                if tokens:
                                    all_desc_tokens.extend([f"{token_type}:{token}" for token in tokens[:3]])
                            token_lists.append(' | '.join(all_desc_tokens))
                    else:
                        # Standard tokenization for other columns
                        for value in self.processed_df[col]:
                            if col == 'company':
                                tokens = self.tokenizer.tokenize_company(value)
                            elif col in ['summary_job_title', 'displayed_job_title']:
                                tokens = self.tokenizer.tokenize_job_title(value)
                            else:
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

            # Create creative job-specific tokenization summary
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
                'creative_job_insights': {
                    'companies_tokenized': len(all_tokens.get('company', [])),
                    'job_titles_tokenized': len(all_tokens.get('summary_job_title', [])) + len(all_tokens.get('displayed_job_title', [])),
                    'descriptions_tokenized': len(all_tokens.get('job_description', [])),
                    'locations_tokenized': len(all_tokens.get('city_job_location', [])) + len(all_tokens.get('state_job_location', [])),
                    'salary_tokens': len(all_tokens.get('job_salary', []))
                }
            }

            st.success("Creative job-specific tokenization completed successfully!")
            logger.info(f"Creative job tokenization completed: {len(valid_stats)} successful columns")
            return True
            
        except Exception as e:
            st.error(f"Error during creative job tokenization: {str(e)}")
            logger.error(f"Creative job tokenization error: {str(e)}")
            return False
    
    def analyze_creative_dataset(self) -> bool:
        """Perform comprehensive creative dataset analysis"""
        if self.processed_df is None:
            st.error("Please process data first")
            return False
        
        try:
            st.info("Performing comprehensive creative job analysis...")
            self.dataset_analysis = analyze_creative_job_dataset(self.processed_df)
            st.success("Creative dataset analysis completed!")
            return True
        except Exception as e:
            st.error(f"Error during dataset analysis: {str(e)}")
            logger.error(f"Dataset analysis error: {str(e)}")
            return False
    
    def build_creative_rag_index(self) -> bool:
        """Build RAG vector index optimized for creative job data"""
        if self.processed_df is None or self.tokenized_df is None:
            st.error("Please process and tokenize creative job data first")
            return False
        
        try:
            st.info("Building creative job-specific RAG vector index...")
            
            # Generate data summary if not already done
            if self.data_summary is None:
                self.generate_creative_data_summary()
            
            # Perform dataset analysis if not done
            if self.dataset_analysis is None:
                self.analyze_creative_dataset()
            
            # Create creative job-specific document chunks
            chunks = self.vector_store.create_document_chunks(self.processed_df, self.data_summary)
            
            # Build vector index
            if chunks and self.vector_store.build_index(chunks):
                st.success("Creative job RAG index built successfully!")
                logger.info("Creative job RAG index built successfully")
                return True
            else:
                st.error("Failed to build creative job RAG index")
                return False
                
        except Exception as e:
            st.error(f"Error building creative job RAG index: {str(e)}")
            logger.error(f"Creative job RAG index building error: {str(e)}")
            return False

    def generate_creative_data_summary(self) -> Optional[Dict]:
        """Generate comprehensive summary optimized for creative job data"""
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
                "creative_job_insights": {},
                "dataset_analysis": self.dataset_analysis if self.dataset_analysis else None
            }

            # Analyze each column with creative job-specific insights
            for col in self.processed_df.columns:
                try:
                    col_info = {
                        "data_type": str(self.processed_df[col].dtype),
                        "null_count": int(self.processed_df[col].isnull().sum()),
                        "unique_count": int(self.processed_df[col].nunique())
                    }

                    # Add creative job-specific analysis
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

            # Generate creative job market insights
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
                
                # Creative-specific insights from dataset analysis
                if self.dataset_analysis:
                    role_analysis = self.dataset_analysis.get('role_analysis', {})
                    adobe_analysis = self.dataset_analysis.get('adobe_analysis', {})
                    ai_analysis = self.dataset_analysis.get('ai_tools_analysis', {})
                    
                    insights['creative_roles'] = {
                        'designer_jobs': role_analysis.get('designer_count', 0),
                        'video_professional_jobs': role_analysis.get('video_professional_count', 0),
                        'photo_professional_jobs': role_analysis.get('photo_professional_count', 0)
                    }
                    
                    insights['software_requirements'] = {
                        'adobe_only_jobs': adobe_analysis.get('adobe_only_count', 0),
                        'non_adobe_only_jobs': adobe_analysis.get('non_adobe_only_count', 0),
                        'both_adobe_and_non_adobe': adobe_analysis.get('both_apps_count', 0),
                        'ai_tools_jobs': ai_analysis.get('ai_tools_count', 0)
                    }
                
                summary['creative_job_insights'] = insights
                
            except Exception as insights_error:
                logger.warning(f"Error generating creative job insights: {str(insights_error)}")
                summary['creative_job_insights'] = {'error': str(insights_error)}

            self.data_summary = summary
            logger.info("Creative job data summary generated successfully")
            return summary
            
        except Exception as e:
            st.error(f"Error generating creative job data summary: {str(e)}")
            logger.error(f"Creative job data summary generation error: {str(e)}")
            return None


def main():
    st.title("Creative Professionals Job Data RAG Analyzer v3.1")
    st.markdown("Upload your creative job dataset CSV and ask intelligent questions with **Enhanced RAG** optimized for **Adobe vs Non-Adobe analysis**, **Creative Roles**, and **AI Tools**!")
    
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
                               help="Enter your OpenAI API key for RAG-enhanced creative job analysis")

        vector_backend_choice = st.selectbox('Vector backend (preference)', options=['usearch','faiss','bruteforce'], index=0)
        if 'creative_processor' in st.session_state and hasattr(st.session_state['creative_processor'], 'vector_store'):
            try:
                st.session_state['creative_processor'].vector_store.backend_preference = vector_backend_choice
            except Exception:
                pass

        st.markdown("---")
        st.header("Creative Analysis Settings")
        
        rag_k_results = st.slider("Retrieved Documents", 5, 20, 10, 
                                 help="Number of relevant creative job documents to retrieve")
        
        maintain_context = st.checkbox("Maintain Conversation Context", value=True,
                                      help="Keep context from previous creative job questions")

    # Check NLTK data
    if not download_nltk_data():
        st.error("Failed to download required NLTK data. Some tokenization features may not work properly.")

    # Initialize session state
    if 'creative_processor' not in st.session_state:
        st.session_state.creative_processor = CreativeJobCSVProcessor()
    if 'creative_openai_processor' not in st.session_state:
        st.session_state.creative_openai_processor = None
    if 'creative_rag_ready' not in st.session_state:
        st.session_state.creative_rag_ready = False

    # Configure OpenAI if API key provided
    if api_key:
        try:
            st.session_state.creative_openai_processor = CreativeJobOpenAIProcessor(api_key)
            if st.session_state.creative_openai_processor.is_available():
                st.sidebar.success("✅ OpenAI API configured")
            else:
                st.sidebar.error(f"❌ OpenAI error: {st.session_state.creative_openai_processor._initialization_error}")
        except Exception as e:
            st.sidebar.error(f"❌ OpenAI error: {str(e)}")
    else:
        st.sidebar.warning("⚠️ Enter OpenAI API key for enhanced creative job analysis")

    # File upload section
    st.header("📁 Upload Creative Job Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file containing creative job data", type="csv",
                                   help="Expected columns: Company, Job Title, Job Description, Location, Salary, etc.")

    if uploaded_file is not None:
        # Load and display basic info
        if st.session_state.creative_processor.load_csv(uploaded_file):
            st.success("✅ Creative job dataset loaded successfully!")

            # Display dataset overview
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Jobs", len(st.session_state.creative_processor.df))
            with col2:
                st.metric("Data Columns", len(st.session_state.creative_processor.df.columns))
            with col3:
                # Try to identify creative job-specific columns
                creative_cols = 0
                for col in st.session_state.creative_processor.df.columns:
                    if any(word in col.lower() for word in ['company', 'job', 'title', 'description', 'location']):
                        creative_cols += 1
                st.metric("Job-Related Columns", creative_cols)

            # Show column information
            with st.expander("📊 Dataset Column Information"):
                col_info = []
                for col in st.session_state.creative_processor.df.columns:
                    col_info.append({
                        'Column': col,
                        'Type': str(st.session_state.creative_processor.df[col].dtype),
                        'Non-Null': f"{st.session_state.creative_processor.df[col].count():,}",
                        'Unique Values': f"{st.session_state.creative_processor.df[col].nunique():,}",
                        'Sample Value': str(st.session_state.creative_processor.df[col].iloc[0])[:50] + "..." if len(str(st.session_state.creative_processor.df[col].iloc[0])) > 50 else str(st.session_state.creative_processor.df[col].iloc[0])
                    })
                st.dataframe(pd.DataFrame(col_info), use_container_width=True)

            # Show sample data
            with st.expander("👀 Sample Creative Job Data"):
                st.dataframe(st.session_state.creative_processor.df.head(10))

            # Processing workflow
            st.header("🔧 Creative Job Data Processing & RAG Setup")
            
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.button("🧹 Clean Data", type="secondary", use_container_width=True):
                    with st.spinner("Cleaning and standardizing creative job data..."):
                        if st.session_state.creative_processor.clean_job_data():
                            st.success("✅ Creative job data cleaned successfully!")
                        else:
                            st.error("❌ Error cleaning creative job data")

            with col2:
                if st.button("🔤 Tokenize Data", type="secondary", use_container_width=True):
                    if st.session_state.creative_processor.processed_df is not None:
                        with st.spinner("Performing creative job-specific tokenization..."):
                            if st.session_state.creative_processor.tokenize_creative_dataset():
                                st.success("✅ Creative job tokenization completed!")
                            else:
                                st.error("❌ Error during creative job tokenization")
                    else:
                        st.error("Please clean data first")

            with col3:
                if st.button("📊 Analyze Dataset", type="secondary", use_container_width=True):
                    if st.session_state.creative_processor.processed_df is not None:
                        with st.spinner("Analyzing creative job patterns..."):
                            if st.session_state.creative_processor.analyze_creative_dataset():
                                st.session_state.creative_processor.generate_creative_data_summary()
                                st.success("✅ Creative dataset analysis completed!")
                            else:
                                st.error("❌ Error during dataset analysis")
                    else:
                        st.error("Please clean data first")

            with col4:
                if st.button("🧠 Build RAG Index", type="primary", use_container_width=True):
                    if (st.session_state.creative_processor.processed_df is not None and 
                        st.session_state.creative_processor.tokenized_df is not None):
                        with st.spinner("Building creative job-specific RAG vector index..."):
                            if st.session_state.creative_processor.build_creative_rag_index():
                                st.session_state.creative_rag_ready = True
                                st.success("✅ Creative job RAG system ready!")
                            else:
                                st.error("❌ Error building creative RAG index")
                    else:
                        st.error("Please clean and tokenize data first")

            # One-click workflow
            st.markdown("### 🚀 Complete Workflow")
            if st.button("🔥 Process Everything (Clean + Tokenize + Analyze + Build RAG)", type="primary", use_container_width=True):
                workflow_success = True
                
                with st.spinner("Running complete creative job processing workflow..."):
                    # Step 1: Clean
                    st.info("Step 1/4: Cleaning creative job data...")
                    if not st.session_state.creative_processor.clean_job_data():
                        st.error("❌ Data cleaning failed")
                        workflow_success = False
                    else:
                        st.success("✅ Data cleaned")
                    
                    # Step 2: Tokenize
                    if workflow_success:
                        st.info("Step 2/4: Tokenizing creative job data...")
                        if st.session_state.creative_processor.tokenize_creative_dataset():
                            st.success("✅ Data tokenized")
                        else:
                            st.error("❌ Tokenization failed")
                            workflow_success = False
                    
                    # Step 3: Analyze
                    if workflow_success:
                        st.info("Step 3/4: Analyzing creative job patterns...")
                        if st.session_state.creative_processor.analyze_creative_dataset():
                            st.session_state.creative_processor.generate_creative_data_summary()
                            st.success("✅ Dataset analyzed")
                        else:
                            st.error("❌ Analysis failed")
                            workflow_success = False
                    
                    # Step 4: Build RAG
                    if workflow_success:
                        st.info("Step 4/4: Building creative RAG index...")
                        if st.session_state.creative_processor.build_creative_rag_index():
                            st.session_state.creative_rag_ready = True
                            st.success("✅ Complete creative RAG system ready!")
                        else:
                            st.error("❌ RAG index building failed")
                            workflow_success = False

                if workflow_success:
                    st.balloons()
                    st.success("🎉 Complete creative job processing workflow completed successfully!")

            # Show processing status
            status_cols = st.columns(4)
            with status_cols[0]:
                if st.session_state.creative_processor.processed_df is not None:
                    st.info("✅ Data Cleaned")
                else:
                    st.warning("⏳ Data Not Cleaned")
            
            with status_cols[1]:
                if st.session_state.creative_processor.tokenized_df is not None:
                    st.info("✅ Data Tokenized")
                else:
                    st.warning("⏳ Data Not Tokenized")
            
            with status_cols[2]:
                if st.session_state.creative_processor.dataset_analysis is not None:
                    st.info("✅ Dataset Analyzed")
                else:
                    st.warning("⏳ Dataset Not Analyzed")
            
            with status_cols[3]:
                if st.session_state.creative_rag_ready:
                    st.info("✅ RAG System Ready")
                else:
                    st.warning("⏳ RAG Not Built")

            # Enhanced analysis section
            if st.session_state.creative_processor.dataset_analysis is not None:
                st.header("🎨 Creative Job Market Analysis & Insights")

                tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🎨 Creative Roles", "💻 Software Analysis", "🤖 AI Tools", "📈 Market Trends"])

                with tab1:
                    st.subheader("Creative Job Dataset Overview")

                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    analysis = st.session_state.creative_processor.dataset_analysis
                    role_analysis = analysis.get('role_analysis', {})
                    adobe_analysis = analysis.get('adobe_analysis', {})
                    ai_analysis = analysis.get('ai_tools_analysis', {})
                    
                    with col1:
                        st.metric("Total Creative Jobs", f"{analysis.get('total_jobs', 0):,}")
                    with col2:
                        st.metric("Designer Roles", f"{role_analysis.get('designer_count', 0):,}")
                    with col3:
                        st.metric("Video Professionals", f"{role_analysis.get('video_professional_count', 0):,}")
                    with col4:
                        st.metric("Photo Professionals", f"{role_analysis.get('photo_professional_count', 0):,}")

                    # Software requirements overview
                    st.subheader("Software Requirements Overview")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Adobe Only", f"{adobe_analysis.get('adobe_only_count', 0):,}")
                    with col2:
                        st.metric("Non-Adobe Only", f"{adobe_analysis.get('non_adobe_only_count', 0):,}")
                    with col3:
                        st.metric("Both Adobe & Non-Adobe", f"{adobe_analysis.get('both_apps_count', 0):,}")
                    with col4:
                        st.metric("AI Tools Required", f"{ai_analysis.get('ai_tools_count', 0):,}")

                    # RAG system status
                    if st.session_state.creative_rag_ready:
                        st.success("🧠 Creative RAG System: Active with creative software-optimized vector search")
                        vector_stats = st.session_state.creative_processor.vector_store.get_stats()
                        st.info(f"📚 Vector Index: {vector_stats['document_count']} creative job documents indexed")
                    else:
                        st.warning("🧠 Creative RAG System: Build index to enable intelligent creative job queries")

                with tab2:
                    st.subheader("Creative Role Analysis")
                    
                    if role_analysis:
                        # Role distribution chart
                        role_counts = {
                            'Designers': role_analysis.get('designer_count', 0),
                            'Video Professionals': role_analysis.get('video_professional_count', 0),
                            'Photo Professionals': role_analysis.get('photo_professional_count', 0)
                        }
                        
                        if any(role_counts.values()):
                            fig = px.pie(
                                values=list(role_counts.values()),
                                names=list(role_counts.keys()),
                                title="Distribution of Creative Professional Roles"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed role information
                        st.write("### Detailed Role Breakdown")
                        
                        if role_analysis.get('designer_jobs'):
                            with st.expander(f"Designer Roles ({len(role_analysis['designer_jobs'])} positions)"):
                                designer_data = []
                                for job in role_analysis['designer_jobs'][:10]:
                                    designer_data.append({
                                        'Title': job.get('title', 'Unknown'),
                                        'Company': job.get('company', 'Unknown'),
                                        'Description Preview': job.get('description_snippet', '')[:100] + "..."
                                    })
                                st.dataframe(pd.DataFrame(designer_data), use_container_width=True)

                with tab3:
                    st.subheader("Creative Software Analysis")
                    
                    if adobe_analysis and analysis.get('software_analysis'):
                        software_analysis = analysis['software_analysis']
                        
                        # Adobe vs Non-Adobe comparison
                        software_comparison = {
                            'Adobe Only': adobe_analysis.get('adobe_only_count', 0),
                            'Non-Adobe Only': adobe_analysis.get('non_adobe_only_count', 0),
                            'Both Adobe & Non-Adobe': adobe_analysis.get('both_apps_count', 0)
                        }
                        
                        fig = px.bar(
                            x=list(software_comparison.keys()),
                            y=list(software_comparison.values()),
                            title="Adobe vs Non-Adobe Software Requirements"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Most mentioned software
                        st.write("### Most Frequently Mentioned Creative Software")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Adobe Applications:**")
                            adobe_freq = software_analysis.get('adobe_apps_frequency', {})
                            if adobe_freq:
                                adobe_df = pd.DataFrame(adobe_freq.most_common(10), columns=['Software', 'Mentions'])
                                st.dataframe(adobe_df, use_container_width=True)
                        
                        with col2:
                            st.write("**Non-Adobe Applications:**")
                            non_adobe_freq = software_analysis.get('non_adobe_apps_frequency', {})
                            if non_adobe_freq:
                                non_adobe_df = pd.DataFrame(non_adobe_freq.most_common(10), columns=['Software', 'Mentions'])
                                st.dataframe(non_adobe_df, use_container_width=True)
                        
                        # Photoshop specific analysis
                        photoshop_count = software_analysis.get('photoshop_count', 0)
                        if photoshop_count > 0:
                            st.metric("Photoshop Mentions", photoshop_count)

                with tab4:
                    st.subheader("AI Tools in Creative Jobs")
                    
                    if ai_analysis and ai_analysis.get('ai_tools_count', 0) > 0:
                        st.metric("Jobs Requiring AI Tools", f"{ai_analysis['ai_tools_count']:,}")
                        
                        # AI tools frequency
                        ai_freq = ai_analysis.get('ai_tools_frequency', {})
                        if ai_freq:
                            ai_df = pd.DataFrame(ai_freq.most_common(10), columns=['AI Tool', 'Mentions'])
                            
                            fig = px.bar(
                                ai_df,
                                x='AI Tool',
                                y='Mentions',
                                title="Most Mentioned AI Tools in Creative Jobs"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.dataframe(ai_df, use_container_width=True)
                        
                        # Sample AI tool jobs
                        if ai_analysis.get('ai_tools_jobs'):
                            with st.expander("Sample Jobs Requiring AI Tools"):
                                ai_jobs_data = []
                                for job in ai_analysis['ai_tools_jobs'][:5]:
                                    ai_jobs_data.append({
                                        'Title': job.get('title', 'Unknown'),
                                        'Company': job.get('company', 'Unknown'),
                                        'AI Tools': ', '.join(job.get('ai_tools', []))
                                    })
                                st.dataframe(pd.DataFrame(ai_jobs_data), use_container_width=True)
                    else:
                        st.info("No AI tools mentioned in the current dataset")

                with tab5:
                    st.subheader("Creative Job Market Trends")
                    
                    # Skills analysis
                    skills_analysis = analysis.get('skills_analysis', {})
                    
                    if skills_analysis:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Most In-Demand Technical Skills:**")
                            tech_skills = skills_analysis.get('technical_skills_frequency', {})
                            if tech_skills:
                                tech_df = pd.DataFrame(tech_skills.most_common(10), columns=['Technical Skill', 'Mentions'])
                                st.dataframe(tech_df, use_container_width=True)
                        
                        with col2:
                            st.write("**Most Mentioned Soft Skills:**")
                            soft_skills = skills_analysis.get('soft_skills_frequency', {})
                            if soft_skills:
                                soft_df = pd.DataFrame(soft_skills.most_common(10), columns=['Soft Skill', 'Mentions'])
                                st.dataframe(soft_df, use_container_width=True)
                        
                        st.write("**Popular Creative Tasks:**")
                        creative_tasks = skills_analysis.get('creative_tasks_frequency', {})
                        if creative_tasks:
                            tasks_df = pd.DataFrame(creative_tasks.most_common(10), columns=['Creative Task', 'Mentions'])
                            st.dataframe(tasks_df, use_container_width=True)

            # RAG-Enhanced Creative Job Query Interface
            if (st.session_state.creative_rag_ready and st.session_state.creative_openai_processor is not None 
                and st.session_state.creative_openai_processor.is_available()):

                st.header("🤖 Intelligent Creative Job Analysis with Enhanced RAG")
                st.markdown("Ask sophisticated questions about **creative professionals**, **Adobe vs non-Adobe software**, **AI tools**, and **creative role requirements**!")

                # RAG system status
                col1, col2, col3 = st.columns(3)
                with col1:
                    vector_stats = st.session_state.creative_processor.vector_store.get_stats()
                    st.info(f"📚 Documents: {vector_stats['document_count']}")
                with col2:
                    st.info(f"🧠 Context: {'Active' if maintain_context else 'Disabled'}")
                with col3:
                    history_count = len(st.session_state.creative_processor.conversation_manager.conversation_history)
                    st.info(f"💬 History: {history_count} exchanges")

                # Example questions from q.txt
                with st.expander("💡 Example Creative Professional Questions (Based on Your q.txt)"):
                    st.markdown("""
                    **Adobe vs Non-Adobe Software Analysis:**
                    - How many postings ask for non-Adobe apps but not Adobe apps? What are those apps?
                    - How many postings ask for both Adobe and non-Adobe apps? What are those app combinations?
                    - How many job listings request experience with Photoshop? And how many request Photoshop's competitors?
                    
                    **Creative Role Analysis:**
                    - How many records describe a designer role? 
                    - Find all designer roles and summarize their creative job requirements
                    - What's the most commonly required creative software in designer roles?
                    - What are the top job titles among designer roles?
                    
                    **Cross-Disciplinary Requirements:**
                    - Which jobs are not video jobs but still require video editing tools? What video tools are they?
                    - Which jobs are not photo jobs but still require photo editing tools? What photo tools are they?
                    - Which jobs are not design jobs but still require design editing tools? What design tools are they?
                    
                    **AI Tools and Modern Creative Workflows:**
                    - How many posts ask for AI skills? What are those AI tools? What are those occupations?
                    - What non-video apps and non-video creative tasks are listed as requirements for video professionals?
                    - What non-photo apps and non-photo creative tasks are listed as requirements for photo professionals?
                    
                    **Industry and Skills Analysis:**
                    - What industries are hiring more creative professionals? What kind of creative professionals?
                    - What are the top mentioned creative activities/tasks?
                    - What soft skills are mentioned in the postings for creative professionals? (collaboration, communication, presentation, etc.)
                    - Can you tell me how many of those jobs ask for Photoshop vs. video editing software? Which is more in demand?
                    """)

                # Query input
                question = st.text_area(
                    "Ask about creative jobs, software requirements, or industry trends:",
                    placeholder="e.g., How many designer roles require Adobe software vs non-Adobe alternatives?",
                    height=100
                )

                # Query options
                col1, col2 = st.columns(2)
                with col1:
                    show_retrieved_docs = st.checkbox("Show Retrieved Context", value=True,
                                                    help="Display creative job data retrieved by RAG")
                with col2:
                    show_conversation_history = st.checkbox("Show Conversation Context", value=False,
                                                          help="Display conversation history")

                if st.button("🚀 Analyze Creative Jobs with RAG", type="primary", use_container_width=True) and question:
                    with st.spinner("Performing intelligent creative job analysis..."):
                        # Process query with enhanced RAG
                        response = st.session_state.creative_openai_processor.query_creative_data_with_rag(
                            question,
                            st.session_state.creative_processor.data_summary,
                            st.session_state.creative_processor.vector_store,
                            st.session_state.creative_processor.conversation_manager,
                            st.session_state.creative_processor.dataset_analysis
                        )

                        # Display results
                        st.subheader("📊 Creative Job Analysis Results")
                        st.write(response)

                        # Show retrieved context if requested
                        if show_retrieved_docs:
                            with st.expander("📄 Retrieved Creative Job Context"):
                                retrieved_docs = st.session_state.creative_processor.vector_store.search(question, k=rag_k_results)
                                if retrieved_docs:
                                    for i, doc in enumerate(retrieved_docs, 1):
                                        similarity = doc['score']
                                        text = doc['text']
                                        metadata = doc.get('metadata', {})
                                        
                                        st.write(f"**Document {i}** (Similarity: {similarity:.3f})")
                                        st.write(text)
                                        
                                        if metadata:
                                            creative_info = []
                                            if metadata.get('company'):
                                                creative_info.append(f"Company: {metadata['company']}")
                                            if metadata.get('location'):
                                                creative_info.append(f"Location: {metadata['location']}")
                                            if metadata.get('adobe_apps'):
                                                creative_info.append(f"Adobe Apps: {', '.join(metadata['adobe_apps'])}")
                                            if metadata.get('non_adobe_apps'):
                                                creative_info.append(f"Non-Adobe Apps: {', '.join(metadata['non_adobe_apps'])}")
                                            if metadata.get('ai_tools'):
                                                creative_info.append(f"AI Tools: {', '.join(metadata['ai_tools'])}")
                                            if metadata.get('is_designer'):
                                                creative_info.append("Role: Designer")
                                            elif metadata.get('is_video_professional'):
                                                creative_info.append("Role: Video Professional")
                                            elif metadata.get('is_photo_professional'):
                                                creative_info.append("Role: Photo Professional")
                                            
                                            if creative_info:
                                                st.caption(" | ".join(creative_info))
                                        
                                        st.markdown("---")
                                else:
                                    st.write("No relevant documents retrieved")

                        # Show conversation context if requested
                        if show_conversation_history and maintain_context:
                            with st.expander("💬 Conversation History"):
                                history = st.session_state.creative_processor.conversation_manager.conversation_history
                                if history:
                                    for i, exchange in enumerate(reversed(history[-5:]), 1):
                                        st.write(f"**Exchange {len(history) - i + 1}** ({exchange.get('query_type', 'general')})")
                                        st.write(f"*Q:* {exchange['question'][:150]}...")
                                        st.write(f"*A:* {exchange['answer'][:200]}...")
                                        st.caption(f"Time: {exchange['timestamp']}")
                                        st.markdown("---")
                                else:
                                    st.write("No conversation history yet")

                # Quick Analysis Buttons for Common Questions
                st.subheader("🔍 Quick Analysis")
                st.markdown("Click for instant analysis of common creative job questions:")
                
                quick_col1, quick_col2, quick_col3 = st.columns(3)
                
                with quick_col1:
                    if st.button("Count Designer Roles", use_container_width=True):
                        if st.session_state.creative_processor.dataset_analysis:
                            designer_count = st.session_state.creative_processor.dataset_analysis['role_analysis'].get('designer_count', 0)
                            st.success(f"**{designer_count}** designer roles found in the dataset")
                        else:
                            st.error("Please analyze dataset first")
                
                with quick_col2:
                    if st.button("Adobe vs Non-Adobe", use_container_width=True):
                        if st.session_state.creative_processor.dataset_analysis:
                            adobe_analysis = st.session_state.creative_processor.dataset_analysis['adobe_analysis']
                            st.success(f"Adobe only: **{adobe_analysis.get('adobe_only_count', 0)}** | Non-Adobe only: **{adobe_analysis.get('non_adobe_only_count', 0)}** | Both: **{adobe_analysis.get('both_apps_count', 0)}**")
                        else:
                            st.error("Please analyze dataset first")
                
                with quick_col3:
                    if st.button("AI Tools Count", use_container_width=True):
                        if st.session_state.creative_processor.dataset_analysis:
                            ai_count = st.session_state.creative_processor.dataset_analysis['ai_tools_analysis'].get('ai_tools_count', 0)
                            st.success(f"**{ai_count}** jobs mention AI tools")
                        else:
                            st.error("Please analyze dataset first")

                # Clear conversation history
                if st.button("🗑️ Clear Conversation History"):
                    st.session_state.creative_processor.conversation_manager.clear_history()
                    st.success("Conversation history cleared")
                    st.rerun()

            elif st.session_state.creative_processor.tokenized_df is not None:
                st.header("⚠️ Enhanced RAG System Setup Required")
                if not st.session_state.creative_openai_processor:
                    st.error("Please enter your OpenAI API key in the sidebar")
                elif not st.session_state.creative_openai_processor.is_available():
                    st.error(f"OpenAI client error: {st.session_state.creative_openai_processor._initialization_error}")
                else:
                    st.warning("Please build the RAG index to enable intelligent creative job queries")

            # Export functionality
            if st.session_state.creative_processor.tokenized_df is not None:
                st.header("📤 Export Enhanced Creative Job Data")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    if st.button("📊 Download Processed Data"):
                        csv = st.session_state.creative_processor.processed_df.to_csv(index=False)
                        st.download_button(
                            label="Download Processed CSV",
                            data=csv,
                            file_name="processed_creative_jobs.csv",
                            mime="text/csv"
                        )

                with col2:
                    if st.button("🔤 Download Tokenized Data"):
                        csv = st.session_state.creative_processor.tokenized_df.to_csv(index=False)
                        st.download_button(
                            label="Download Tokenized CSV",
                            data=csv,
                            file_name="tokenized_creative_jobs.csv",
                            mime="text/csv"
                        )

                with col3:
                    if st.button("📚 Download RAG Documents"):
                        if st.session_state.creative_processor.vector_store.documents:
                            rag_data = {
                                "documents": st.session_state.creative_processor.vector_store.documents,
                                "metadata": st.session_state.creative_processor.vector_store.metadata,
                                "total_documents": len(st.session_state.creative_processor.vector_store.documents),
                                "vector_dimension": st.session_state.creative_processor.vector_store.dimension
                            }
                            json_data = json.dumps(rag_data, indent=2)
                            st.download_button(
                                label="Download RAG Documents JSON",
                                data=json_data,
                                file_name="creative_jobs_rag_documents.json",
                                mime="application/json"
                            )

                with col4:
                    if st.button("📈 Download Analysis Results"):
                        if st.session_state.creative_processor.dataset_analysis:
                            analysis_json = json.dumps(st.session_state.creative_processor.dataset_analysis, indent=2, default=str)
                            st.download_button(
                                label="Download Analysis JSON",
                                data=analysis_json,
                                file_name="creative_jobs_analysis.json",
                                mime="application/json"
                            )

    # Footer
    st.markdown("---")
    st.markdown("""
    ## Creative Professionals Job Data RAG Analyzer v3.1 - Complete Solution

    **Enhanced Features for Creative Job Analysis:**
    - **Adobe vs Non-Adobe Analysis**: Comprehensive software requirement analysis
    - **Creative Role Categorization**: Automatic detection of designers, video professionals, photo specialists
    - **AI Tools Integration**: Detection and analysis of AI tool requirements in creative workflows
    - **Enhanced RAG System**: Vector search optimized for creative software and skills
    - **Professional Analytics**: Detailed insights for creative industry trends

    **Perfect for answering questions like:**
    - How many designer roles require specific software combinations?
    - Which creative jobs are adopting AI tools?
    - What's the demand for Adobe vs alternative creative software?
    - Cross-disciplinary skill requirements analysis
    - Creative industry hiring patterns and trends

    **Dependencies for Full Functionality:**
    ```bash
    pip install streamlit pandas numpy openai plotly nltk tiktoken usearch sentence-transformers
    ```
    
    **How to Run:**
    ```bash
    # Save all parts as creative_job_rag_analyzer_v31.py, then run:
    streamlit run creative_job_rag_analyzer_v31.py
    ```
    
    **Optimized for Creative Professionals Analysis:**
    - HR professionals in creative industries
    - Creative recruiters and talent acquisition
    - Creative professionals planning career transitions
    - Market researchers analyzing creative job trends
    - Educational institutions developing creative programs
    """)

if __name__ == "__main__":
    main()


