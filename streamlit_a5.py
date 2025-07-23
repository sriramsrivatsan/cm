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
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import string
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('words', quiet=True)
        return True
    except:
        return False

# Set page config
st.set_page_config(
    page_title="CSV Natural Language Query App with Advanced Tokenization",
    page_icon="📊",
    layout="wide"
)

class AdvancedTokenizer:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def tokenize_text(self, text: str, method='lemmatize') -> List[str]:
        """Advanced text tokenization with multiple options"""
        if pd.isna(text) or not isinstance(text, str):
            return []

        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)

        # Tokenize into words
        tokens = word_tokenize(text)

        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]

        # Apply stemming or lemmatization
        if method == 'stem':
            tokens = [self.stemmer.stem(token) for token in tokens]
        elif method == 'lemmatize':
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        # Remove empty tokens
        tokens = [token for token in tokens if token.strip()]

        return tokens

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
        """Tokenize categorical values with frequency context"""
        if pd.isna(value) or value == 'Unknown':
            return ['missing_category', 'unknown']

        tokens = []

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

        return tokens

class CSVProcessor:
    def __init__(self):
        self.df = None
        self.processed_df = None
        self.tokenized_df = None
        self.data_summary = None
        self.tokenization_summary = None
        self.tokenizer = AdvancedTokenizer()

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
                col_info["top_values"] = top_values.to_dict()

                # Add tokenization info if available
                if self.tokenization_summary and col in self.tokenization_summary['column_stats']:
                    col_info['tokenization'] = self.tokenization_summary['column_stats'][col]

            summary["column_details"][col] = col_info

        self.data_summary = summary
        return summary

class OpenAIQueryProcessor:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)

    def generate_system_prompt(self, data_summary: Dict, include_tokenization: bool = True) -> str:
        """Generate an enhanced system prompt based on the data summary and tokenization"""
        prompt = f"""You are an advanced data analyst assistant with access to a comprehensively tokenized dataset.

Dataset Overview:
- Total rows: {data_summary['dataset_info']['total_rows']}
- Total columns: {data_summary['dataset_info']['total_columns']}
- Columns: {', '.join(data_summary['dataset_info']['column_names'])}

"""

        if include_tokenization and data_summary.get('tokenization_info'):
            token_info = data_summary['tokenization_info']
            prompt += f"""
Tokenization Summary:
- Total tokens generated: {token_info['global_stats']['total_tokens_generated']:,}
- Unique tokens: {token_info['global_stats']['total_unique_tokens']:,}
- Average tokens per column: {token_info['global_stats']['average_tokens_per_column']:.1f}
- Token diversity score: {token_info['global_stats']['average_diversity_per_column']:.2f}

"""

        prompt += "Detailed Column Analysis:\n"

        for col, details in data_summary['column_details'].items():
            prompt += f"\n{col}:"
            prompt += f"\n  - Type: {details['data_type']}"
            prompt += f"\n  - Unique values: {details['unique_count']}"

            if 'min' in details:
                prompt += f"\n  - Range: {details['min']} to {details['max']}"
                prompt += f"\n  - Mean: {details['mean']:.2f}"

            if 'top_values' in details:
                top_vals = list(details['top_values'].keys())[:3]
                prompt += f"\n  - Common values: {', '.join(map(str, top_vals))}"

            # Add tokenization details
            if 'tokenization' in details:
                token_stats = details['tokenization']
                prompt += f"\n  - Tokens generated: {token_stats['total_tokens']}"
                prompt += f"\n  - Unique tokens: {token_stats['unique_tokens']}"
                prompt += f"\n  - Token diversity: {token_stats['token_diversity']:.2f}"
                if token_stats['most_common']:
                    top_tokens = [f"{token}({count})" for token, count in token_stats['most_common'][:3]]
                    prompt += f"\n  - Top tokens: {', '.join(top_tokens)}"

        prompt += """

Enhanced Capabilities with Tokenization:
1. Each data point has been extensively tokenized to capture semantic meaning, context, and patterns
2. Numeric values include magnitude, sign, and contextual tokens
3. Dates include temporal context (seasons, decades, relative time)
4. Categories include frequency context and semantic tokenization
5. All tokens are searchable and queryable for complex natural language questions

When answering questions:
1. Leverage the rich tokenization to provide deeper insights
2. Use token patterns to identify trends and correlations
3. Reference token diversity scores to assess data complexity
4. Suggest analyses based on available token types
5. Provide specific, actionable insights based on the tokenized structure
6. Mention relevant token patterns when explaining findings
"""

        return prompt

    def query_data(self, question: str, data_summary: Dict, sample_data: str = None, tokenized_sample: str = None) -> str:
        """Process natural language query about the tokenized data"""
        system_prompt = self.generate_system_prompt(data_summary, include_tokenization=True)

        user_message = f"Question: {question}"
        if sample_data:
            user_message += f"\n\nSample original data (first 5 rows):\n{sample_data}"
        if tokenized_sample:
            user_message += f"\n\nSample tokenized data (showing token patterns):\n{tokenized_sample}"

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=1500,
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error processing query: {str(e)}"

def main():
    st.title("📊 Advanced CSV Natural Language Query App")
    st.markdown("Upload a CSV file and ask questions about your data with **comprehensive tokenization** for enhanced natural language understanding!")

    # Check NLTK data
    if not download_nltk_data():
        st.error("Failed to download required NLTK data. Some tokenization features may not work properly.")

    # Initialize session state
    if 'processor' not in st.session_state:
        st.session_state.processor = CSVProcessor()
    if 'openai_processor' not in st.session_state:
        st.session_state.openai_processor = None

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
            st.header("🔧 Data Processing & Tokenization")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Clean Data", type="secondary"):
                    with st.spinner("Cleaning data..."):
                        if st.session_state.processor.clean_data():
                            st.success("✅ Data cleaned successfully!")
                        else:
                            st.error("❌ Error cleaning data")

            with col2:
                if st.button("🚀 Clean + Tokenize Data", type="primary"):
                    with st.spinner("Processing and tokenizing data..."):
                        # First clean the data
                        if st.session_state.processor.clean_data():
                            st.success("✅ Data cleaned!")

                            # Then tokenize
                            if st.session_state.processor.tokenize_dataset():
                                st.session_state.processor.generate_data_summary()
                                st.balloons()
                                st.success("🎉 Data processing and tokenization completed!")
                            else:
                                st.error("❌ Error during tokenization")
                        else:
                            st.error("❌ Error cleaning data")

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

                    # Data type distribution
                    st.subheader("Data Type Distribution")
                    dtype_counts = st.session_state.processor.processed_df.dtypes.value_counts()

                    col1, col2 = st.columns(2)
                    with col1:
                        fig_dtype = px.pie(values=dtype_counts.values, names=dtype_counts.index,
                                          title="Column Data Types")
                        st.plotly_chart(fig_dtype, use_container_width=True)

                    with col2:
                        st.write("**Data Type Breakdown:**")
                        for dtype, count in dtype_counts.items():
                            percentage = (count / len(st.session_state.processor.processed_df.columns)) * 100
                            st.write(f"• {dtype}: {count} columns ({percentage:.1f}%)")

                    # Dataset quality metrics
                    st.subheader("Data Quality Metrics")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        completeness = ((st.session_state.processor.processed_df.size - missing_values) / st.session_state.processor.processed_df.size) * 100
                        st.metric("Data Completeness", f"{completeness:.2f}%")

                    with col2:
                        # Calculate uniqueness (average percentage of unique values per column)
                        uniqueness_scores = []
                        for col in st.session_state.processor.processed_df.columns:
                            unique_pct = (st.session_state.processor.processed_df[col].nunique() / len(st.session_state.processor.processed_df)) * 100
                            uniqueness_scores.append(unique_pct)
                        avg_uniqueness = np.mean(uniqueness_scores)
                        st.metric("Avg Uniqueness", f"{avg_uniqueness:.2f}%")

                    with col3:
                        # Variability score (columns with more than 1 unique value)
                        variable_cols = sum(1 for col in st.session_state.processor.processed_df.columns
                                          if st.session_state.processor.processed_df[col].nunique() > 1)
                        variability = (variable_cols / len(st.session_state.processor.processed_df.columns)) * 100
                        st.metric("Data Variability", f"{variability:.2f}%")

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

                        # Token efficiency metrics
                        st.write("### ⚡ Tokenization Efficiency")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            # Token-to-data ratio
                            total_data_points = st.session_state.processor.processed_df.size
                            token_ratio = token_summary['global_stats']['total_tokens_generated'] / total_data_points
                            st.metric("Tokens per Data Point", f"{token_ratio:.2f}")

                        with col2:
                            # Compression ratio (unique tokens vs total tokens)
                            compression = (token_summary['global_stats']['total_unique_tokens'] /
                                         token_summary['global_stats']['total_tokens_generated']) * 100
                            st.metric("Token Compression", f"{compression:.1f}%")

                        with col3:
                            # Semantic richness (unique tokens per column)
                            semantic_richness = token_summary['global_stats']['total_unique_tokens'] / len(token_summary['column_stats'])
                            st.metric("Semantic Richness", f"{semantic_richness:.1f}")

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
                            fig_tokens.update_xaxis(tickangle=45)
                            st.plotly_chart(fig_tokens, use_container_width=True)

                        with col2:
                            # Diversity score distribution
                            diversity_scores = [token_summary['column_stats'][col]['token_diversity'] for col in columns]

                            fig_diversity = px.bar(x=columns, y=diversity_scores,
                                                 title="Token Diversity by Column",
                                                 labels={'x': 'Columns', 'y': 'Diversity Score'},
                                                 color=diversity_scores,
                                                 color_continuous_scale='viridis')
                            fig_diversity.update_xaxis(tickangle=45)
                            st.plotly_chart(fig_diversity, use_container_width=True)

                        # Token frequency heatmap
                        st.write("### 🔥 Token Frequency Heatmap")

                        # Create a matrix of top tokens per column
                        top_tokens_matrix = []
                        all_top_tokens = set()

                        # Collect all top tokens
                        for col, stats in token_summary['column_stats'].items():
                            for token, count in stats['most_common'][:10]:
                                all_top_tokens.add(token)

                        # Create frequency matrix
                        if all_top_tokens:
                            matrix_data = []
                            for token in list(all_top_tokens)[:20]:  # Limit to top 20 for readability
                                row = [token]
                                for col in columns:
                                    # Find token frequency in this column
                                    token_freq = 0
                                    for t, count in token_summary['column_stats'][col]['most_common']:
                                        if t == token:
                                            token_freq = count
                                            break
                                    row.append(token_freq)
                                matrix_data.append(row)

                            if matrix_data:
                                heatmap_df = pd.DataFrame(matrix_data, columns=['Token'] + columns)
                                heatmap_df = heatmap_df.set_index('Token')

                                fig_heatmap = px.imshow(heatmap_df.values,
                                                      x=heatmap_df.columns,
                                                      y=heatmap_df.index,
                                                      title="Token Frequency Across Columns",
                                                      color_continuous_scale='blues',
                                                      aspect='auto')
                                fig_heatmap.update_layout(height=600)
                                st.plotly_chart(fig_heatmap, use_container_width=True)

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
                        for i, col in enumerate(numeric_cols[:4]):  # Show first 4 numeric columns
                            col1, col2 = st.columns(2)

                            with col1:
                                fig_hist = px.histogram(st.session_state.processor.processed_df, x=col,
                                                      title=f"Distribution of {col}")
                                st.plotly_chart(fig_hist, use_container_width=True)

                            with col2:
                                fig_box = px.box(st.session_state.processor.processed_df, y=col,
                                               title=f"Box Plot of {col}")
                                st.plotly_chart(fig_box, use_container_width=True)

                    # Categorical columns analysis
                    cat_cols = st.session_state.processor.processed_df.select_dtypes(include=['object']).columns
                    if len(cat_cols) > 0:
                        st.write("### 🏷️ Categorical Columns Distribution")

                        for col in cat_cols[:4]:  # Show first 4 categorical columns
                            if st.session_state.processor.processed_df[col].nunique() <= 20:
                                value_counts = st.session_state.processor.processed_df[col].value_counts().head(10)

                                col1, col2 = st.columns(2)

                                with col1:
                                    fig_bar = px.bar(x=value_counts.index, y=value_counts.values,
                                                   title=f"Top Values in {col}")
                                    fig_bar.update_xaxis(tickangle=45)
                                    st.plotly_chart(fig_bar, use_container_width=True)

                                with col2:
                                    fig_pie = px.pie(values=value_counts.values, names=value_counts.index,
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
                                fig_tokens = px.bar(token_df, x='Token', y='Frequency',
                                                  title=f"Top Tokens in {selected_column}")
                                fig_tokens.update_xaxis(tickangle=45)
                                st.plotly_chart(fig_tokens, use_container_width=True)

                        # Data distribution for selected column
                        st.write("### 📈 Data Distribution")

                        if col_data.dtype in ['int64', 'float64']:
                            # Numeric column analysis
                            col1, col2 = st.columns(2)

                            with col1:
                                st.write("**Statistical Summary:**")
                                stats = col_data.describe()
                                for stat, value in stats.items():
                                    st.write(f"• {stat}: {value:.3f}")

                            with col2:
                                # Quartile information
                                q1 = col_data.quantile(0.25)
                                q3 = col_data.quantile(0.75)
                                iqr = q3 - q1
                                outliers = col_data[(col_data < q1 - 1.5*iqr) | (col_data > q3 + 1.5*iqr)]

                                st.write("**Quartile Analysis:**")
                                st.write(f"• Q1: {q1:.3f}")
                                st.write(f"• Q3: {q3:.3f}")
                                st.write(f"• IQR: {iqr:.3f}")
                                st.write(f"• Outliers: {len(outliers)} ({len(outliers)/len(col_data)*100:.1f}%)")

                            # Distribution plots
                            fig_hist = px.histogram(col_data, title=f"Distribution of {selected_column}",
                                                  marginal="box")
                            st.plotly_chart(fig_hist, use_container_width=True)

                        else:
                            # Categorical column analysis
                            value_counts = col_data.value_counts()

                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Value Frequency:**")
                                for value, count in value_counts.head(10).items():
                                    percentage = (count / len(col_data)) * 100
                                    st.write(f"• {value}: {count} ({percentage:.1f}%)")

                            with col2:
                                st.write("**Category Statistics:**")
                                st.write(f"• Most common: {value_counts.index[0]} ({value_counts.iloc[0]} occurrences)")
                                st.write(f"• Least common: {value_counts.index[-1]} ({value_counts.iloc[-1]} occurrences)")
                                st.write(f"• Mode frequency: {value_counts.iloc[0]/len(col_data)*100:.1f}%")

                            if len(value_counts) <= 20:
                                fig_bar = px.bar(x=value_counts.index, y=value_counts.values,
                                               title=f"Value Distribution in {selected_column}")
                                fig_bar.update_xaxis(tickangle=45)
                                st.plotly_chart(fig_bar, use_container_width=True)

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

                # Detailed tokenization analysis
                if show_token_details and st.session_state.processor.tokenization_summary:
                    with st.expander("📊 Detailed Tokenization Analysis"):
                        for col, stats in st.session_state.processor.tokenization_summary['column_stats'].items():
                            st.subheader(f"Column: {col}")

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Tokens", stats['total_tokens'])
                            with col2:
                                st.metric("Unique Tokens", stats['unique_tokens'])
                            with col3:
                                st.metric("Diversity Score", f"{stats['token_diversity']:.3f}")

                            if stats['most_common']:
                                st.write("**Most Common Tokens:**")
                                for token, count in stats['most_common'][:10]:
                                    st.write(f"• {token}: {count}")

                            st.markdown("---")

            # Natural Language Query Section
            if (st.session_state.processor.tokenized_df is not None and
                st.session_state.openai_processor is not None):

                st.header("🤖 Enhanced Natural Language Queries")
                st.markdown("Ask sophisticated questions about your **tokenized** data!")

                # Enhanced example questions
                with st.expander("💡 Enhanced Example Questions"):
                    st.markdown("""
                    **Dataset Statistics & Overview:**
                    - What are the key statistical insights from this dataset?
                    - Which columns have the highest data quality and why?
                    - How does the tokenization enhance understanding of data patterns?
                    - What is the distribution of data types and what does it suggest about the dataset?

                    **Tokenization-Based Analysis:**
                    - Which columns have the highest token diversity and what insights does this provide?
                    - What are the most frequent semantic tokens and what patterns do they reveal?
                    - How do token patterns correlate with data quality metrics?
                    - What unique insights emerge from the token frequency analysis?

                    **Advanced Pattern Recognition:**
                    - Based on statistical analysis, what are the most important variables for analysis?
                    - What hidden relationships are revealed through cross-column token analysis?
                    - How do the data distribution patterns inform potential modeling approaches?
                    - What anomalies or outliers are suggested by the statistical and token analysis?

                    **Business Intelligence:**
                    - What actionable insights can be derived from the comprehensive data statistics?
                    - How should the statistical patterns influence data preprocessing decisions?
                    - What do the token patterns suggest about the underlying business processes?
                    """)

                # Query input with statistics context
                question = st.text_area("Ask a sophisticated question about your tokenized data and statistics:",
                                       placeholder="e.g., Based on the statistical analysis and tokenization, what are the key drivers and patterns in this dataset?",
                                       height=100)

                if st.button("🔍 Analyze with Enhanced Tokenization & Statistics") and question:
                    with st.spinner("Performing comprehensive analysis on tokenized data and statistics..."):
                        # Prepare comprehensive context including statistics
                        sample_data = st.session_state.processor.processed_df.head(3).to_string()

                        # Prepare tokenized sample
                        token_columns = [col for col in st.session_state.processor.tokenized_df.columns if col.endswith('_tokens')]
                        if token_columns:
                            tokenized_sample = st.session_state.processor.tokenized_df[token_columns].head(3).to_string()
                        else:
                            tokenized_sample = None

                        # Add statistical context
                        stats_context = ""
                        if st.session_state.processor.tokenization_summary:
                            stats_context = f"""

Statistical Context:
- Dataset Size: {len(st.session_state.processor.processed_df):,} rows × {len(st.session_state.processor.processed_df.columns)} columns
- Memory Usage: {st.session_state.processor.processed_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
- Missing Values: {st.session_state.processor.processed_df.isnull().sum().sum():,}
- Data Completeness: {((st.session_state.processor.processed_df.size - st.session_state.processor.processed_df.isnull().sum().sum()) / st.session_state.processor.processed_df.size) * 100:.2f}%

Tokenization Statistics:
- Total Tokens Generated: {st.session_state.processor.tokenization_summary['global_stats']['total_tokens_generated']:,}
- Unique Semantic Tokens: {st.session_state.processor.tokenization_summary['global_stats']['total_unique_tokens']:,}
- Average Token Diversity: {st.session_state.processor.tokenization_summary['global_stats']['average_diversity_per_column']:.3f}
- Token Compression Ratio: {(st.session_state.processor.tokenization_summary['global_stats']['total_unique_tokens'] / st.session_state.processor.tokenization_summary['global_stats']['total_tokens_generated']) * 100:.1f}%
"""

                        # Get AI response with enhanced context
                        enhanced_question = question + stats_context
                        response = st.session_state.openai_processor.query_data(
                            enhanced_question,
                            st.session_state.processor.data_summary,
                            sample_data,
                            tokenized_sample
                        )

                        st.subheader("🎯 Comprehensive Analysis Results")
                        st.write(response)

                        # Add quick statistical insights
                        if st.session_state.processor.tokenization_summary:
                            with st.expander("📊 Related Statistical Insights"):
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.write("**Data Quality Indicators:**")
                                    completeness = ((st.session_state.processor.processed_df.size - st.session_state.processor.processed_df.isnull().sum().sum()) / st.session_state.processor.processed_df.size) * 100
                                    st.write(f"• Data Completeness: {completeness:.2f}%")

                                    # Calculate uniqueness scores
                                    uniqueness_scores = []
                                    for col in st.session_state.processor.processed_df.columns:
                                        unique_pct = (st.session_state.processor.processed_df[col].nunique() / len(st.session_state.processor.processed_df)) * 100
                                        uniqueness_scores.append(unique_pct)
                                    avg_uniqueness = np.mean(uniqueness_scores)
                                    st.write(f"• Average Uniqueness: {avg_uniqueness:.2f}%")

                                    # Most diverse column
                                    token_stats = st.session_state.processor.tokenization_summary['column_stats']
                                    most_diverse = max(token_stats.keys(), key=lambda x: token_stats[x]['token_diversity'])
                                    st.write(f"• Most Semantically Rich Column: {most_diverse}")

                                with col2:
                                    st.write("**Tokenization Insights:**")
                                    token_ratio = st.session_state.processor.tokenization_summary['global_stats']['total_tokens_generated'] / st.session_state.processor.processed_df.size
                                    st.write(f"• Tokens per Data Point: {token_ratio:.2f}")

                                    semantic_richness = st.session_state.processor.tokenization_summary['global_stats']['total_unique_tokens'] / len(token_stats)
                                    st.write(f"• Semantic Richness Score: {semantic_richness:.1f}")

                                    # Least diverse column
                                    least_diverse = min(token_stats.keys(), key=lambda x: token_stats[x]['token_diversity'])
                                    st.write(f"• Most Structured Column: {least_diverse}")

                # Statistics Dashboard
                st.header("📊 Interactive Statistics Dashboard")

                dashboard_tab1, dashboard_tab2, dashboard_tab3 = st.tabs(["📈 Key Metrics", "🎯 Data Quality", "🔄 Correlations"])

                with dashboard_tab1:
                    st.subheader("Key Dataset Metrics")

                    # Create a comprehensive metrics table
                    metrics_data = []

                    for col in st.session_state.processor.processed_df.columns:
                        col_data = st.session_state.processor.processed_df[col]

                        metric_row = {
                            'Column': col,
                            'Data Type': str(col_data.dtype),
                            'Count': len(col_data),
                            'Missing': col_data.isnull().sum(),
                            'Missing %': f"{(col_data.isnull().sum() / len(col_data)) * 100:.1f}%",
                            'Unique': col_data.nunique(),
                            'Unique %': f"{(col_data.nunique() / len(col_data)) * 100:.1f}%"
                        }

                        # Add type-specific metrics
                        if col_data.dtype in ['int64', 'float64']:
                            metric_row.update({
                                'Mean': f"{col_data.mean():.3f}",
                                'Std': f"{col_data.std():.3f}",
                                'Min': f"{col_data.min():.3f}",
                                'Max': f"{col_data.max():.3f}"
                            })
                        else:
                            mode_val = col_data.mode().iloc[0] if not col_data.mode().empty else 'N/A'
                            metric_row.update({
                                'Mode': str(mode_val)[:20] + '...' if len(str(mode_val)) > 20 else str(mode_val),
                                'Mode Freq': col_data.value_counts().iloc[0] if not col_data.value_counts().empty else 0
                            })

                        # Add tokenization metrics if available
                        if (st.session_state.processor.tokenization_summary and
                            col in st.session_state.processor.tokenization_summary['column_stats']):
                            token_stats = st.session_state.processor.tokenization_summary['column_stats'][col]
                            metric_row.update({
                                'Tokens': token_stats['total_tokens'],
                                'Unique Tokens': token_stats['unique_tokens'],
                                'Token Diversity': f"{token_stats['token_diversity']:.3f}"
                            })

                        metrics_data.append(metric_row)

                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df, use_container_width=True, height=400)

                    # Download metrics report
                    csv_metrics = metrics_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Metrics Report",
                        data=csv_metrics,
                        file_name="dataset_metrics_report.csv",
                        mime="text/csv"
                    )

                with dashboard_tab2:
                    st.subheader("Data Quality Assessment")

                    # Overall quality score
                    completeness = ((st.session_state.processor.processed_df.size - st.session_state.processor.processed_df.isnull().sum().sum()) / st.session_state.processor.processed_df.size) * 100

                    # Calculate consistency score (based on data types and patterns)
                    consistency_scores = []
                    for col in st.session_state.processor.processed_df.columns:
                        col_data = st.session_state.processor.processed_df[col]
                        if col_data.dtype == 'object':
                            # For object columns, consistency is based on pattern regularity
                            if col_data.nunique() < len(col_data) * 0.1:  # Low cardinality suggests consistent categories
                                consistency_scores.append(0.9)
                            else:
                                consistency_scores.append(0.6)
                        else:
                            # For numeric columns, consistency is based on outlier ratio
                            q1 = col_data.quantile(0.25)
                            q3 = col_data.quantile(0.75)
                            iqr = q3 - q1
                            outliers = col_data[(col_data < q1 - 1.5*iqr) | (col_data > q3 + 1.5*iqr)]
                            outlier_ratio = len(outliers) / len(col_data)
                            consistency_scores.append(max(0.3, 1.0 - outlier_ratio * 2))

                    consistency = np.mean(consistency_scores) * 100

                    # Uniqueness score
                    uniqueness_scores = []
                    for col in st.session_state.processor.processed_df.columns:
                        unique_pct = st.session_state.processor.processed_df[col].nunique() / len(st.session_state.processor.processed_df)
                        uniqueness_scores.append(unique_pct)
                    uniqueness = np.mean(uniqueness_scores) * 100

                    # Overall quality score
                    overall_quality = (completeness * 0.4 + consistency * 0.4 + uniqueness * 0.2)

                    # Display quality metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Overall Quality", f"{overall_quality:.1f}/100",
                                 delta=f"{'Excellent' if overall_quality > 80 else 'Good' if overall_quality > 60 else 'Needs Improvement'}")
                    with col2:
                        st.metric("Completeness", f"{completeness:.1f}%")
                    with col3:
                        st.metric("Consistency", f"{consistency:.1f}%")
                    with col4:
                        st.metric("Uniqueness", f"{uniqueness:.1f}%")

                    # Quality breakdown by column
                    st.write("### Column-wise Quality Assessment")

                    quality_data = []
                    for col in st.session_state.processor.processed_df.columns:
                        col_data = st.session_state.processor.processed_df[col]

                        # Column completeness
                        col_completeness = ((len(col_data) - col_data.isnull().sum()) / len(col_data)) * 100

                        # Column uniqueness
                        col_uniqueness = (col_data.nunique() / len(col_data)) * 100

                        # Column consistency
                        if col_data.dtype == 'object':
                            # Check for consistent formatting/patterns
                            if col_data.nunique() < len(col_data) * 0.1:
                                col_consistency = 90
                            elif col_data.nunique() < len(col_data) * 0.5:
                                col_consistency = 70
                            else:
                                col_consistency = 50
                        else:
                            # Check for outliers
                            q1 = col_data.quantile(0.25)
                            q3 = col_data.quantile(0.75)
                            iqr = q3 - q1
                            outliers = col_data[(col_data < q1 - 1.5*iqr) | (col_data > q3 + 1.5*iqr)]
                            outlier_ratio = len(outliers) / len(col_data)
                            col_consistency = max(30, (1.0 - outlier_ratio * 2) * 100)

                        # Overall column quality
                        col_quality = (col_completeness * 0.4 + col_consistency * 0.4 + col_uniqueness * 0.2)

                        quality_data.append({
                            'Column': col,
                            'Quality Score': f"{col_quality:.1f}/100",
                            'Completeness': f"{col_completeness:.1f}%",
                            'Consistency': f"{col_consistency:.1f}%",
                            'Uniqueness': f"{col_uniqueness:.1f}%",
                            'Issues': 'Low Completeness' if col_completeness < 90 else
                                     'Low Consistency' if col_consistency < 70 else
                                     'Low Uniqueness' if col_uniqueness < 10 else 'None'
                        })

                    quality_df = pd.DataFrame(quality_data)
                    st.dataframe(quality_df, use_container_width=True)

                    # Quality visualization
                    fig_quality = px.bar(quality_df, x='Column', y='Quality Score',
                                        title="Data Quality Score by Column",
                                        color='Quality Score',
                                        color_continuous_scale='RdYlGn')
                    fig_quality.update_xaxis(tickangle=45)
                    fig_quality.update_traces(texttemplate='%{y}', textposition='outside')
                    st.plotly_chart(fig_quality, use_container_width=True)

                with dashboard_tab3:
                    st.subheader("Correlation & Relationship Analysis")

                    # Numeric correlations
                    numeric_cols = st.session_state.processor.processed_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 1:
                        st.write("### Numeric Variable Correlations")

                        corr_matrix = st.session_state.processor.processed_df[numeric_cols].corr()

                        # Correlation heatmap
                        fig_corr = px.imshow(corr_matrix,
                                           title="Correlation Matrix of Numeric Variables",
                                           color_continuous_scale='RdBu',
                                           aspect='auto')
                        fig_corr.update_layout(height=600)
                        st.plotly_chart(fig_corr, use_container_width=True)

                        # Strong correlations
                        st.write("### Strong Correlations (|r| > 0.7)")
                        strong_corrs = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                corr_val = corr_matrix.iloc[i, j]
                                if abs(corr_val) > 0.7:
                                    strong_corrs.append({
                                        'Variable 1': corr_matrix.columns[i],
                                        'Variable 2': corr_matrix.columns[j],
                                        'Correlation': f"{corr_val:.3f}",
                                        'Strength': 'Very Strong' if abs(corr_val) > 0.9 else 'Strong'
                                    })

                        if strong_corrs:
                            strong_corr_df = pd.DataFrame(strong_corrs)
                            st.dataframe(strong_corr_df, use_container_width=True)
                        else:
                            st.info("No strong correlations (|r| > 0.7) found between numeric variables.")

                    # Token-based relationships
                    if st.session_state.processor.tokenization_summary:
                        st.write("### Token-based Relationship Insights")

                        # Find columns with similar token patterns
                        token_stats = st.session_state.processor.tokenization_summary['column_stats']

                        # Calculate token diversity similarity
                        diversity_similarities = []
                        columns = list(token_stats.keys())

                        for i in range(len(columns)):
                            for j in range(i+1, len(columns)):
                                col1, col2 = columns[i], columns[j]
                                div1 = token_stats[col1]['token_diversity']
                                div2 = token_stats[col2]['token_diversity']

                                similarity = 1 - abs(div1 - div2)  # Simple similarity measure

                                if similarity > 0.8:  # High similarity
                                    diversity_similarities.append({
                                        'Column 1': col1,
                                        'Column 2': col2,
                                        'Diversity Similarity': f"{similarity:.3f}",
                                        'Column 1 Diversity': f"{div1:.3f}",
                                        'Column 2 Diversity': f"{div2:.3f}"
                                    })

                        if diversity_similarities:
                            st.write("**Columns with Similar Token Diversity Patterns:**")
                            similarity_df = pd.DataFrame(diversity_similarities)
                            st.dataframe(similarity_df, use_container_width=True)

                        # Most vs least diverse columns comparison
                        most_diverse = max(token_stats.keys(), key=lambda x: token_stats[x]['token_diversity'])
                        least_diverse = min(token_stats.keys(), key=lambda x: token_stats[x]['token_diversity'])

                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"**Most Semantically Complex:** {most_diverse}")
                            st.write(f"Diversity Score: {token_stats[most_diverse]['token_diversity']:.3f}")
                            st.write(f"Total Tokens: {token_stats[most_diverse]['total_tokens']:,}")
                            st.write("*Best for: Complex pattern analysis, rich semantic queries*")

                        with col2:
                            st.info(f"**Most Structured/Predictable:** {least_diverse}")
                            st.write(f"Diversity Score: {token_stats[least_diverse]['token_diversity']:.3f}")
                            st.write(f"Total Tokens: {token_stats[least_diverse]['total_tokens']:,}")
                            st.write("*Best for: Classification, grouping, categorical analysis*")

                # Advanced Statistics Summary
                st.header("📋 Executive Statistics Summary")

                if st.button("📊 Generate Comprehensive Statistics Report"):
                    with st.spinner("Generating comprehensive statistics report..."):

                        # Create comprehensive report
                        report_sections = []

                        # Dataset overview
                        report_sections.append("## 📊 Dataset Overview")
                        report_sections.append(f"- **Size**: {len(st.session_state.processor.processed_df):,} rows × {len(st.session_state.processor.processed_df.columns)} columns")
                        report_sections.append(f"- **Memory Usage**: {st.session_state.processor.processed_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                        report_sections.append(f"- **Missing Values**: {st.session_state.processor.processed_df.isnull().sum().sum():,} ({(st.session_state.processor.processed_df.isnull().sum().sum() / st.session_state.processor.processed_df.size) * 100:.2f}%)")

                        # Data types breakdown
                        dtype_counts = st.session_state.processor.processed_df.dtypes.value_counts()
                        report_sections.append("\n## 🏷️ Data Types Distribution")
                        for dtype, count in dtype_counts.items():
                            percentage = (count / len(st.session_state.processor.processed_df.columns)) * 100
                            report_sections.append(f"- **{dtype}**: {count} columns ({percentage:.1f}%)")

                        # Tokenization summary
                        if st.session_state.processor.tokenization_summary:
                            token_summary = st.session_state.processor.tokenization_summary
                            report_sections.append("\n## 🏷️ Tokenization Statistics")
                            report_sections.append(f"- **Total Tokens Generated**: {token_summary['global_stats']['total_tokens_generated']:,}")
                            report_sections.append(f"- **Unique Semantic Tokens**: {token_summary['global_stats']['total_unique_tokens']:,}")
                            report_sections.append(f"- **Average Token Diversity**: {token_summary['global_stats']['average_diversity_per_column']:.3f}")
                            report_sections.append(f"- **Tokens per Data Point**: {token_summary['global_stats']['total_tokens_generated'] / st.session_state.processor.processed_df.size:.2f}")

                        # Quality assessment
                        completeness = ((st.session_state.processor.processed_df.size - st.session_state.processor.processed_df.isnull().sum().sum()) / st.session_state.processor.processed_df.size) * 100
                        report_sections.append("\n## ✅ Data Quality Assessment")
                        report_sections.append(f"- **Overall Completeness**: {completeness:.2f}%")
                        report_sections.append(f"- **Quality Status**: {'Excellent' if completeness > 95 else 'Good' if completeness > 85 else 'Acceptable' if completeness > 75 else 'Needs Attention'}")

                        # Top insights
                        if st.session_state.processor.tokenization_summary:
                            token_stats = st.session_state.processor.tokenization_summary['column_stats']
                            most_diverse = max(token_stats.keys(), key=lambda x: token_stats[x]['token_diversity'])
                            least_diverse = min(token_stats.keys(), key=lambda x: token_stats[x]['token_diversity'])

                            report_sections.append("\n## 🎯 Key Insights")
                            report_sections.append(f"- **Most Semantically Rich Column**: {most_diverse} (diversity: {token_stats[most_diverse]['token_diversity']:.3f})")
                            report_sections.append(f"- **Most Structured Column**: {least_diverse} (diversity: {token_stats[least_diverse]['token_diversity']:.3f})")

                            # Find columns with most tokens
                            highest_token_col = max(token_stats.keys(), key=lambda x: token_stats[x]['total_tokens'])
                            report_sections.append(f"- **Most Tokenized Column**: {highest_token_col} ({token_stats[highest_token_col]['total_tokens']:,} tokens)")

                        # Recommendations
                        report_sections.append("\n## 💡 Recommendations")
                        if completeness < 85:
                            report_sections.append("- **Data Quality**: Consider addressing missing values before analysis")
                        if st.session_state.processor.tokenization_summary:
                            avg_diversity = st.session_state.processor.tokenization_summary['global_stats']['average_diversity_per_column']
                            if avg_diversity > 0.7:
                                report_sections.append("- **High Diversity**: Dataset has rich semantic content, excellent for complex NLP queries")
                            elif avg_diversity < 0.3:
                                report_sections.append("- **Low Diversity**: Dataset is highly structured, ideal for classification and pattern recognition")

                        # Display the report
                        full_report = "\n".join(report_sections)
                        st.markdown(full_report)

                        # Download report
                        st.download_button(
                            label="📥 Download Statistics Report",
                            data=full_report,
                            file_name="comprehensive_statistics_report.md",
                            mime="text/markdown"
                        )

                # Token Search Interface
                st.subheader("🔍 Token Search & Analysis")
                if st.session_state.processor.tokenized_df is not None:
                    search_token = st.text_input("Search for specific tokens across all columns:")

                    if search_token:
                        with st.spinner("Searching tokens..."):
                            # Search across all token columns
                            token_columns = [col for col in st.session_state.processor.tokenized_df.columns if col.endswith('_tokens')]

                            matches = {}
                            for col in token_columns:
                                matching_rows = st.session_state.processor.tokenized_df[
                                    st.session_state.processor.tokenized_df[col].str.contains(search_token, case=False, na=False)
                                ].index.tolist()

                                if matching_rows:
                                    matches[col] = len(matching_rows)

                            if matches:
                                st.success(f"Found '{search_token}' in {len(matches)} columns!")

                                # Display matches
                                for col, count in matches.items():
                                    original_col = col.replace('_tokens', '')
                                    st.write(f"**{original_col}**: {count} matches")

                                # Show sample matches
                                if st.button("Show Sample Matches"):
                                    for col in list(matches.keys())[:3]:  # Show first 3 columns
                                        original_col = col.replace('_tokens', '')
                                        matching_rows = st.session_state.processor.tokenized_df[
                                            st.session_state.processor.tokenized_df[col].str.contains(search_token, case=False, na=False)
                                        ]

                                        if not matching_rows.empty:
                                            st.write(f"**Sample matches in {original_col}:**")
                                            st.dataframe(matching_rows[[original_col, col]].head(3))
                            else:
                                st.warning(f"No matches found for '{search_token}'")

                # Quick stats section
                st.header("📈 Enhanced Quick Statistics")
                if st.button("Generate Token-Enhanced Stats"):
                    df = st.session_state.processor.processed_df

                    # Numeric columns stats with token context
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        st.subheader("Numeric Columns with Token Analysis")
                        stats_df = df[numeric_cols].describe()
                        st.dataframe(stats_df)

                        # Add token-based insights
                        for col in numeric_cols[:3]:  # Show first 3 numeric columns
                            if st.session_state.processor.tokenization_summary:
                                col_tokens = st.session_state.processor.tokenization_summary['column_stats'].get(col, {})
                                if col_tokens:
                                    st.write(f"**{col} Token Insights:**")
                                    st.write(f"- Generated {col_tokens['total_tokens']} contextual tokens")
                                    st.write(f"- Token diversity: {col_tokens['token_diversity']:.3f}")

                                    # Show distribution with token context
                                    fig = px.histogram(df, x=col, title=f"Distribution of {col} (with {col_tokens['unique_tokens']} unique tokens)")
                                    st.plotly_chart(fig, use_container_width=True)

                    # Categorical columns with enhanced token analysis
                    cat_cols = df.select_dtypes(include=['object']).columns
                    if len(cat_cols) > 0:
                        st.subheader("Categorical Columns with Token Enhancement")
                        for col in cat_cols[:3]:  # Show first 3 categorical columns
                            st.write(f"**{col}** - {df[col].nunique()} unique values")

                            # Show token statistics if available
                            if st.session_state.processor.tokenization_summary:
                                col_tokens = st.session_state.processor.tokenization_summary['column_stats'].get(col, {})
                                if col_tokens:
                                    st.write(f"- Generated {col_tokens['total_tokens']} semantic tokens")
                                    st.write(f"- Token diversity: {col_tokens['token_diversity']:.3f}")
                                    st.write(f"- Most common tokens: {', '.join([token for token, count in col_tokens['most_common'][:5]])}")

                            if df[col].nunique() <= 15:
                                fig = px.bar(x=df[col].value_counts().index,
                                           y=df[col].value_counts().values,
                                           title=f"Distribution of {col}")
                                st.plotly_chart(fig, use_container_width=True)

                # Token Pattern Analysis
                st.header("🧠 Token Pattern Analysis")
                if st.button("Analyze Token Patterns"):
                    if st.session_state.processor.tokenization_summary:
                        token_summary = st.session_state.processor.tokenization_summary

                        # Create visualizations of token statistics
                        columns = list(token_summary['column_stats'].keys())
                        token_counts = [token_summary['column_stats'][col]['total_tokens'] for col in columns]
                        diversity_scores = [token_summary['column_stats'][col]['token_diversity'] for col in columns]

                        # Token count distribution
                        fig1 = px.bar(x=columns, y=token_counts,
                                     title="Token Count by Column",
                                     labels={'x': 'Columns', 'y': 'Token Count'})
                        fig1.update_xaxis(tickangle=45)
                        st.plotly_chart(fig1, use_container_width=True)

                        # Token diversity scores
                        fig2 = px.bar(x=columns, y=diversity_scores,
                                     title="Token Diversity Score by Column",
                                     labels={'x': 'Columns', 'y': 'Diversity Score'},
                                     color=diversity_scores,
                                     color_continuous_scale='viridis')
                        fig2.update_xaxis(tickangle=45)
                        st.plotly_chart(fig2, use_container_width=True)

                        # Overall insights
                        st.subheader("🎯 Key Tokenization Insights")

                        # Find most and least diverse columns
                        most_diverse = max(columns, key=lambda x: token_summary['column_stats'][x]['token_diversity'])
                        least_diverse = min(columns, key=lambda x: token_summary['column_stats'][x]['token_diversity'])

                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"**Most Diverse Column:** {most_diverse}")
                            st.write(f"Diversity Score: {token_summary['column_stats'][most_diverse]['token_diversity']:.3f}")
                            st.write("This column has the richest variety of semantic tokens, indicating complex patterns.")

                        with col2:
                            st.info(f"**Least Diverse Column:** {least_diverse}")
                            st.write(f"Diversity Score: {token_summary['column_stats'][least_diverse]['token_diversity']:.3f}")
                            st.write("This column has more repetitive patterns, suitable for classification tasks.")

                        # Token frequency analysis
                        st.subheader("🔥 Global Token Frequency Analysis")
                        all_tokens = []
                        for col_stats in token_summary['column_stats'].values():
                            all_tokens.extend([token for token, count in col_stats['most_common']])

                        global_token_freq = Counter(all_tokens)
                        most_common_global = global_token_freq.most_common(20)

                        if most_common_global:
                            tokens, counts = zip(*most_common_global)
                            fig3 = px.bar(x=list(tokens), y=list(counts),
                                         title="Top 20 Most Frequent Tokens Across All Columns",
                                         labels={'x': 'Tokens', 'y': 'Frequency'})
                            fig3.update_xaxis(tickangle=45)
                            st.plotly_chart(fig3, use_container_width=True)

                            st.write("**Interpretation:** These tokens appear frequently across multiple columns and represent key semantic concepts in your dataset.")

    # Export functionality
    if st.session_state.processor.tokenized_df is not None:
        st.header("📥 Export Enhanced Data")

        col1, col2 = st.columns(2)

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

        # Export tokenization summary
        if st.session_state.processor.tokenization_summary:
            if st.button("📋 Download Tokenization Report"):
                report = json.dumps(st.session_state.processor.tokenization_summary, indent=2)
                st.download_button(
                    label="Download Tokenization JSON Report",
                    data=report,
                    file_name="tokenization_report.json",
                    mime="application/json"
                )

    # Footer with advanced tips
    st.markdown("---")
    st.markdown("""
    ## 🚀 Advanced Tokenization Features

    **What makes this tokenization special:**
    - **Contextual Numeric Tokens**: Numbers include magnitude, sign, and range information
    - **Temporal Intelligence**: Dates are tokenized with seasons, decades, and relative time context
    - **Semantic Categories**: Text categories are broken down into meaningful semantic components
    - **Frequency Context**: Common vs rare values are identified and tokenized accordingly
    - **Column Awareness**: All tokens include column context for better querying

    **Best Practices:**
    - Use specific, detailed questions to leverage the rich tokenization
    - Search for semantic tokens to find patterns not visible in raw data
    - Compare token diversity scores to understand data complexity
    - Use the token search feature to find cross-column relationships

    💡 **Pro Tip**: The more detailed your natural language questions, the better the AI can utilize the comprehensive tokenization to provide insights!
    """)

if __name__ == "__main__":
    main()
