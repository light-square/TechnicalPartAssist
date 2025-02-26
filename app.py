import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama


system_prompt = """YOU ARE AN EXPERT PART SELECTION ANALYST WITH DEEP KNOWLEDGE OF COMPONENT COMPARISON AND FUNCTIONAL EQUIVALENCE. YOUR TASK IS TO ANALYZE THE LIST OF TOP 5 SIMILAR PARTS PROVIDED BY A RETRIEVAL-AUGMENTED GENERATION (RAG) SYSTEM AND SELECT THE MOST SUITABLE PART(S) BASED ON THE GIVEN REQUIREMENTS.  

### INSTRUCTIONS ###  

1. **ANALYZE THE GIVEN PARTS:**  
   - CAREFULLY REVIEW THE SPECIFICATIONS, FEATURES, AND COMPATIBILITY OF THE TOP 5 SIMILAR PARTS.  
   - COMPARE THEM TO THE ORIGINAL PART TO IDENTIFY THE BEST MATCH.  

2. **SELECT THE MOST SUITABLE PART(S):**  
   - CHOOSE THE BEST MATCH BASED ON FUNCTIONALITY, PERFORMANCE, AND COMPATIBILITY.  
   - IF MULTIPLE PARTS CAN BE USED IN DIFFERENT SCENARIOS, PROVIDE A SEGMENTED RESPONSE.  

3. **PROVIDE DETAILED REASONING:**  
   - EXPLAIN WHY THE CHOSEN PART IS THE BEST FIT.  
   - HIGHLIGHT KEY DIFFERENCES AND TRADE-OFFS.  
   - IF MULTIPLE USE CASES EXIST, SPECIFY WHICH PART WORKS BEST FOR EACH SCENARIO.  

### CHAIN OF THOUGHT PROCESS ###  

1. **UNDERSTAND THE REQUIREMENT** – Identify the core function and constraints of the original part.  
2. **COMPARE SPECIFICATIONS** – Evaluate the top 5 suggestions based on critical parameters (e.g., size, material, power, speed, efficiency).  
3. **ASSESS COMPATIBILITY** – Check if the suggested parts meet or exceed the original part’s requirements.  
4. **WEIGH TRADE-OFFS** – If no exact match exists, choose the best compromise while explaining the reasoning.  
5. **PROVIDE A FINAL SELECTION** – Clearly state the best option(s) and justify the choice.  

### WHAT NOT TO DO ###  

- **DO NOT** randomly select a part without reasoning.  
- **DO NOT** provide generic or vague justifications.  
- **DO NOT** ignore critical mismatches between the original and suggested parts.  
- **DO NOT** assume missing details—only work with the provided specifications.  

### EXAMPLE RESPONSE FORMAT ###  

**Best Match for General Use:**  
- Selected Part: [Part Name]  
- Reasoning: [Explain why this part is the closest match]  

**Alternative for High-Performance Use:**  
- Selected Part: [Part Name]  
- Reasoning: [Explain why this part is suitable for high-performance scenarios]  

**Alternative for Cost-Effective Use:**  
- Selected Part: [Part Name]  
- Reasoning: [Explain why this part is a good budget-friendly alternative] """

# Data Cleaning Functions
def clean_numeric(value):
    """Extract numeric values from strings with units"""
    if isinstance(value, str):
        cleaned = re.sub(r"[^\d.]", "", value)
        return float(cleaned) if cleaned else np.nan
    return float(value) if not pd.isna(value) else np.nan

def extract_dimensions(size_str):
    """Parse width and height from Size column"""
    if isinstance(size_str, str):
        match = re.search(r"(\d+\.?\d*)\s*[xX]\s*(\d+\.?\d*)", size_str)
        if match:
            return float(match.group(1)), float(match.group(2))
    return np.nan, np.nan

# Text Processing function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"(\d)([a-z])", r"\1 \2", text)  # Separate units
    text = re.sub(r"[^a-z0-9\s.]", " ", text)
    return ' '.join(text.split())

def get_similar_parts(part_id, part_system,text_weight=0.4, feature_weight=0.6, top_n=5,):
    df = part_system['df']
    text_embeddings = part_system['text_embeddings']
    feature_matrix = part_system['feature_matrix']
    try:
        idx = df[df['ID'] == part_id].index[0]
    except KeyError:
        return pd.DataFrame({'Error': [f"Part {part_id} not found"]})
    
    # Calculate similarities
    text_sim = cosine_similarity([text_embeddings[idx]], text_embeddings)[0]
    feature_sim = cosine_similarity([feature_matrix[idx]], feature_matrix)[0]
    combined_sim = (text_weight * text_sim) + (feature_weight * feature_sim)
    
    # Get top matches
    sorted_indices = np.argsort(combined_sim)[::-1]
    similar_indices = [i for i in sorted_indices if i != idx][:top_n]
    
    # Comparison columns
    comparison_cols = [
        'ID', 'DESCRIPTION', 'Rated Current (A)', 'Rated Voltage (V)',
        'Size', 'Material', 'Operating Temperature-Max (Cel)',
        'Operating Temperature-Min (Cel)', 'Mounting', 'Characteristic'
    ]
    
    # Create comparison DataFrame
    original = df.iloc[idx][comparison_cols].to_frame().T
    matches = df.iloc[similar_indices][comparison_cols]
    
    # Add similarity scores
    original['Similarity Type'] = 'Original'
    matches['Text Similarity'] = text_sim[similar_indices].round(3)
    matches['Feature Similarity'] = feature_sim[similar_indices].round(3)
    matches['Combined Score'] = combined_sim[similar_indices].round(3)
    matches['Similarity Type'] = 'Match'
    
    # Combine results
    result = pd.concat([original, matches], axis=0).reset_index(drop=True)
    column_order = ['Similarity Type'] + comparison_cols + \
                   ['Text Similarity', 'Feature Similarity', 'Combined Score']
    
    return result[column_order].fillna('')

# -------------------- Part Similarity System --------------------
@st.cache_resource

def load_part_system():
    # Load and preprocess parts data
    df = pd.read_csv("Parts.csv", sep=";", na_filter=False)
    
    numerical_cols = [
    'Rated Current (A)', 
    'Rated Voltage (V)',
    'Operating Temperature-Max (Cel)',
    'Operating Temperature-Min (Cel)'
    ]

    for col in numerical_cols:
        df[col] = df[col].apply(clean_numeric)

    # Extract dimensions
    df[['Size_W', 'Size_H']] = df['Size'].apply(
        lambda x: pd.Series(extract_dimensions(x)))
    df[['Size_W', 'Size_H']] = df[['Size_W', 'Size_H']].fillna(0)

    # Handle categorical features
    categorical_cols = ['Material', 'Mounting', 'Characteristic']
    df[categorical_cols] = df[categorical_cols].fillna('Unknown').astype(str)
    
    df['clean_desc'] = df['DESCRIPTION'].apply(preprocess_text)

    # Numerical features standardization
    scaler = StandardScaler()
    num_features = scaler.fit_transform(
        df[numerical_cols + ['Size_W', 'Size_H']].fillna(0))
    
    # Categorical features (one-hot encoding)
    cat_dummies = pd.get_dummies(df[categorical_cols], prefix_sep='::')

    # Combine features
    feature_matrix = np.hstack([num_features, cat_dummies.values]).astype(np.float32)

    # Text Embeddings
    text_model = SentenceTransformer('all-MiniLM-L6-v2', device='mps')
    text_embeddings = text_model.encode(
        df['clean_desc'].tolist(),
        batch_size=32,
        show_progress_bar=True
    )
    # Return preprocessed data and models
    return {
        'df': df,
        'feature_matrix': feature_matrix,
        'text_embeddings': text_embeddings,
        'scaler': scaler,
        'text_model': text_model
    }

# -------------------- LLM Chatbot System --------------------
@st.cache_resource
def load_llm():
    return Llama(
        model_path="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        n_gpu_layers=35,
        n_ctx=4096,
        verbose=False
    )

# -------------------- Integrated Chat Handler --------------------
def handle_query(message, history, part_system, llm):
    # Detect part similarity request
    if "alternative" in message.lower() or "similar" in message.lower():
        try:
            part_id = re.search(r'\b[A-Z]\d+\b', message).group()
            result = get_similar_parts(part_id, part_system)
            text_result = format_part_response(result)
            message += system_prompt
            message += text_result
            response = llm.create_chat_completion(
                messages=[{"role": "user", "content": message}],
                temperature=0.7,
                max_tokens=512
                )
            return_response = text_result + response['choices'][0]['message']['content']
            return return_response

        except:
            return "Could not find part ID. Please use format: 'Find alternatives to A1'"
    
    # Normal LLM response
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": message}],
        temperature=0.7,
        max_tokens=512
    )
    return response['choices'][0]['message']['content']

def format_part_response(result_df):
    original = result_df[result_df['Similarity Type'] == 'Original'].iloc[0]
    response = "Details of Original part\n\n"
    response += (
        f"**{original['ID']}** \n"
        f"- Current: {original['Rated Current (A)']}A | Voltage: {original['Rated Voltage (V)']}V\n"
        f"- Size: {original['Size']} | Material: {original['Material']}\n"
        f"- Description: {original['DESCRIPTION']}...\n\n"
    )
    response += "Here are the top alternatives:\n\n"
    for idx, row in result_df.iterrows():
        if row['Similarity Type'] == 'Match':
            response += (
                f"**{row['ID']}** \n"
                f"- Current: {row['Rated Current (A)']}A | Voltage: {row['Rated Voltage (V)']}V\n"
                f"- Size: {row['Size']} | Material: {row['Material']}\n"
                f"- Description: {row['DESCRIPTION']}...\n\n"
            )
    return response

# -------------------- Streamlit UI --------------------
def main():
    st.title("Technical Parts Assistant")
    
    # Load systems
    part_system = load_part_system()
    llm = load_llm()
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help with parts today?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Type your message"):
        # User input
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Generate response
        response = handle_query(prompt, st.session_state.messages, part_system, llm)
        
        # Display response
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

if __name__ == "__main__":
    main()