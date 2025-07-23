import streamlit as st
import pandas as pd
import numpy as np
import openai
from io import StringIO
import json
import re
from typing import Dict, List, Any
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="CSV Natural Language Query App",
    page_icon="📊",
    layout="wide"
)

class CSVProcessor:
    def __init__(self):
        self.df = None
        self.processed_df = None
        self.data_summary = None

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
                    pd.to_datetime(self.processed_df[col], infer_datetime_format=True)
                    self.processed_df[col] = pd.to_datetime(self.processed_df[col])
                except:
                    # Try to convert to numeric
                    try:
                        self.processed_df[col] = pd.to_numeric(self.processed_df[col])
                    except:
                        pass  # Keep as string

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
            "column_details": {}
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
            elif self.processed_df[col].dtype == 'object':
                # Get top 5 most common values
                top_values = self.processed_df[col].value_counts().head(5)
                col_info["top_values"] = top_values.to_dict()

            summary["column_details"][col] = col_info

        self.data_summary = summary
        return summary

class OpenAIQueryProcessor:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)

    def generate_system_prompt(self, data_summary: Dict) -> str:
        """Generate a system prompt based on the data summary"""
        prompt = f"""You are a data analyst assistant. You have access to a dataset with the following structure:

Dataset Overview:
- Total rows: {data_summary['dataset_info']['total_rows']}
- Total columns: {data_summary['dataset_info']['total_columns']}
- Columns: {', '.join(data_summary['dataset_info']['column_names'])}

Column Details:
"""

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

        prompt += """

When answering questions about this data:
1. Provide specific insights based on the data structure
2. Suggest relevant visualizations when appropriate
3. If asked for analysis, provide statistical insights
4. If the question requires specific data values, mention that you'd need to query the actual dataset
5. Always be clear about what analysis is possible with this data structure
"""

        return prompt

    def query_data(self, question: str, data_summary: Dict, sample_data: str = None) -> str:
        """Process natural language query about the data"""
        system_prompt = self.generate_system_prompt(data_summary)

        user_message = f"Question: {question}"
        if sample_data:
            user_message += f"\n\nSample data (first 5 rows):\n{sample_data}"

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=1000,
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error processing query: {str(e)}"

def main():
    st.title("📊 CSV Natural Language Query App")
    st.markdown("Upload a CSV file and ask questions about your data in natural language!")

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
            st.header("🔧 Data Processing")

            if st.button("Clean and Process Data"):
                with st.spinner("Processing data..."):
                    if st.session_state.processor.clean_data():
                        st.session_state.processor.generate_data_summary()
                        st.success("✅ Data processed successfully!")

                        # Show processed data info
                        st.subheader("Processed Data Overview")

                        # Display data summary
                        summary = st.session_state.processor.data_summary

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Processed Rows", summary['dataset_info']['total_rows'])
                        with col2:
                            st.metric("Processed Columns", summary['dataset_info']['total_columns'])
                        with col3:
                            missing_data = sum([details['null_count']
                                              for details in summary['column_details'].values()])
                            st.metric("Missing Values Handled", missing_data)

                        # Show processed data sample
                        with st.expander("View Processed Data Sample"):
                            st.dataframe(st.session_state.processor.processed_df.head())

                        # Show column information
                        with st.expander("Column Information"):
                            for col, details in summary['column_details'].items():
                                st.write(f"**{col}**")
                                st.write(f"- Type: {details['data_type']}")
                                st.write(f"- Unique values: {details['unique_count']}")
                                if 'min' in details:
                                    st.write(f"- Range: {details['min']} to {details['max']}")
                                if 'top_values' in details and details['top_values']:
                                    top_3 = list(details['top_values'].items())[:3]
                                    st.write(f"- Top values: {', '.join([f'{k} ({v})' for k, v in top_3])}")
                                st.write("---")

            # Natural Language Query Section
            if (st.session_state.processor.processed_df is not None and
                st.session_state.openai_processor is not None):

                st.header("🤖 Natural Language Queries")
                st.markdown("Ask questions about your data in plain English!")

                # Example questions
                with st.expander("💡 Example Questions"):
                    st.markdown("""
                    - What are the main patterns in this data?
                    - Which columns have the most variation?
                    - What kind of analysis would be most valuable for this dataset?
                    - Are there any correlations I should be aware of?
                    - What visualizations would help understand this data better?
                    - Summarize the key insights from this dataset
                    """)

                # Query input
                question = st.text_area("Ask a question about your data:",
                                       placeholder="e.g., What are the main trends in this data?")

                if st.button("🔍 Analyze") and question:
                    with st.spinner("Analyzing your data..."):
                        # Prepare sample data for context
                        sample_data = st.session_state.processor.processed_df.head().to_string()

                        # Get AI response
                        response = st.session_state.openai_processor.query_data(
                            question,
                            st.session_state.processor.data_summary,
                            sample_data
                        )

                        st.subheader("📋 Analysis Results")
                        st.write(response)

                # Quick stats section
                st.header("📈 Quick Statistics")
                if st.button("Generate Quick Stats"):
                    df = st.session_state.processor.processed_df

                    # Numeric columns stats
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        st.subheader("Numeric Columns Statistics")
                        st.dataframe(df[numeric_cols].describe())

                    # Categorical columns info
                    cat_cols = df.select_dtypes(include=['object']).columns
                    if len(cat_cols) > 0:
                        st.subheader("Categorical Columns")
                        for col in cat_cols[:5]:  # Show first 5 categorical columns
                            st.write(f"**{col}** - {df[col].nunique()} unique values")
                            if df[col].nunique() <= 10:
                                fig = px.bar(x=df[col].value_counts().index,
                                           y=df[col].value_counts().values,
                                           title=f"Distribution of {col}")
                                st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("💡 **Tip**: Make sure your CSV file has proper headers and is well-formatted for best results!")

if __name__ == "__main__":
    main()
