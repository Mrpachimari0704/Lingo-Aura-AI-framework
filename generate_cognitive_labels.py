# 1_generate_cognitive_labels.py
# ==============================================================================
#  Standalone script to generate cognitive-linguistic labels for CMU-MOSEI.
# ==============================================================================
#
# HOW TO RUN:
# 1. Make sure your OpenAI API key is set in the CONFIG section.
# 2. Make sure you have downloaded the required .csd files and placed them in
#    the `data/cmumosei/` folder, specifically:
#    - CMU_MOSEI_TimestampedWords.csd
#    - CMU_MOSEI_Labels.csd
# 3. Run this script from your project's root directory:
#    python 1_generate_cognitive_labels.py
#
# This will create the `cmu_mosei_with_cognitive_labels.csv` file, which is
# required by the main training script.

import os
import json
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# Use mmsdk just to parse the .csd files
from mmsdk import mmdatasdk as md

# --- CONFIGURATION ---
# IMPORTANT: Replace "sk-YOUR_API_KEY" with your actual OpenAI API key.
DEEPSEEK_API_KEY = "sk-2982f8f6a23243df85824edb16f2973f"


DATA_PATH = "./data/cmumosei/"
OUTPUT_CSV_PATH = os.path.join(DATA_PATH, "cmu_mosei_with_cognitive_labels_v4.csv")

# PROMPT_TEMPLATE_FOR_API = """
# You are a linguistics expert. Based on the transcript and the expressed emotion of a speaker,
# annotate their cognitive-linguistic state using the provided JSON schema.
# Only output a valid JSON object.

# Schema:
# {{
#     "Reasoning Mode": ["Inductive", "Deductive", "Analogical", "None"],
#     "Information Stance": ["Retrieving", "Stating Opinion", "Confirming Fact", "Questioning"],
#     "Affective Expression": ["Direct", "Subtle", "Suppressed", "Ironic"],
#     "Social Intent": ["Seeking Empathy", "Debating", "Relationship Maintenance", "Ending Conversation"]
# }}

# Transcript: "{transcription}"
# Emotion Score: {emotion_score:.2f} (from -3 to +3, where positive is happy/excited)

# Your JSON Annotation:
# """


PROMPT_TEMPLATE_FOR_API = """
You are an expert in linguistics and psychology. Your task is to analyze a transcript and infer the speaker's cognitive-linguistic state.

Follow these steps carefully:
1.  **Analyze the text**: Read the transcript and understand its core message, logical structure, and emotional tone.
2.  **Infer each dimension**: For each of the four dimensions below, choose the most fitting category based on your analysis.
3.  **Format the output**: Provide your final answer ONLY as a valid JSON object.

---
**Dimension Definitions:**

*   **"Reasoning Mode"**: How is the speaker thinking?
    *   `"Deductive"`: Applying a general rule to a specific case (e.g., "All men are mortal, Socrates is a man, therefore Socrates is mortal"). Defining categories.
    *   `"Inductive"`: Deriving a general rule from specific observations.
    *   `"Analogical"`: Making a comparison between two different things (e.g., "Life is like a box of chocolates").
    *   `"None"`: No clear logical reasoning is present.

*   **"Information Stance"**: What is the speaker doing with information?
    *   `"Stating Opinion"`: Expressing a personal belief or judgment.
    *   `"Confirming Fact"`: Stating something as an objective truth.
    *   `"Questioning"`: Asking for information or expressing doubt.
    *   `"Retrieving"`: Recalling a memory or past event.

*   **"Affective Expression"**: How is emotion being expressed?
    *   `"Direct"`: Stating feelings explicitly (e.g., "I am happy").
    *   `"Subtle"`: Hinting at feelings through word choice or tone, not stated directly.
    *   `"Suppressed"`: Actively trying to hide or downplay emotions.
    *   `"Ironic"`: Saying the opposite of what is meant.

*   **"Social Intent"**: What is the goal of the communication?
    *   `"Information Transfer"`: The primary goal is to teach, explain, or inform.
    *   `"Persuasion"`: Trying to convince the listener of something.
    *   `"Relationship Maintenance"`: Building rapport, apologizing, expressing empathy.
    *   `"Ending Conversation"`: Signaling a desire to stop talking.

---
**Example:**

*   **Transcript**: "i see that there are three category of writers i define them as being an author a writer and a story teller..."
*   **Emotion Score**: 1.00
*   **Your Analysis (Internal Thought Process)**: The speaker is defining categories and explaining his personal framework. This is a form of deductive reasoning and the main goal is to inform.
*   **Your JSON Annotation**:
    ```json
    {{
        "Reasoning Mode": ["Deductive"],
        "Information Stance": ["Stating Opinion"],
        "Affective Expression": ["Direct"],
        "Social Intent": ["Information Transfer"]
    }}
    ```

---
**Now, analyze the following new case:**

Transcript: "{transcription}"
Emotion Score: {emotion_score:.2f}

Your JSON Annotation:
"""

# --- MAIN SCRIPT LOGIC ---
def main():
    """
    Main function to load data, call the DeepSeek API, and save results.
    """
    print("="*60)
    print("Cognitive-Linguistic Label Generation for CMU-MOSEI (using DeepSeek)")
    print("="*60)

    # --- ✨ 核心修改 2: 初始化客户端时指向 DeepSeek 服务器 ✨ ---
    try:
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1"  # 这是关键！
        )
        # 测试 API Key 和连接
        client.models.list() 
    except Exception as e:
        print("\nFATAL ERROR: Could not initialize DeepSeek client.")
        print("Please check if your DEEPSEEK_API_KEY is correct and if you have network access.")
        print(f"Error details: {e}")
        return

    if os.path.exists(OUTPUT_CSV_PATH):
        print(f"Found existing progress file: {OUTPUT_CSV_PATH}. Loading and resuming.")
        df = pd.read_csv(OUTPUT_CSV_PATH)
    else:
        print("No existing progress file found. Extracting data from .csd files...")
        try:
            text_field = 'CMU_MOSEI_TimestampedWords'
            label_field = 'CMU_MOSEI_Labels'
            recipe = {
                text_field: os.path.join(DATA_PATH, text_field) + '.csd',
                label_field: os.path.join(DATA_PATH, label_field) + '.csd'
            }
            dataset = md.mmdataset(recipe)
        except RuntimeError as e:
            print(f"\nFATAL ERROR: Could not load .csd files. Error: {e}")
            return
            
        all_data = []
        for segment_id, data in tqdm(dataset[label_field].data.items(), desc="Extracting text and labels"):
            text_features = dataset[text_field].data.get(segment_id, {}).get('features')
            if text_features is None: continue
            words = [word[0].decode('utf-8') for word in text_features if word[0] != b'sp']
            sentence = " ".join(words)
            label = data['features'][0][0]
            all_data.append({'segment_id': segment_id, 'text': sentence, 'emotion_score': label})
        
        df = pd.DataFrame(all_data)
        df['cognitive_label'] = None

    print("\nStarting annotation process with DeepSeek...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Annotating samples"):
        if pd.notna(row['cognitive_label']) and row['cognitive_label'] != '{}':
            continue

        prompt = PROMPT_TEMPLATE_FOR_API.format(
            transcription=row['text'],
            emotion_score=row['emotion_score']
        )
        
        success = False
        for attempt in range(3):
            try:
                # --- ✨ 核心修改 3: 使用 DeepSeek 的模型名称 ✨ ---
                response = client.chat.completions.create(
                    model="deepseek-chat",  # 使用官方推荐的强大聊天模型
                    messages=[{"role": "user", "content": prompt}],
                    # DeepSeek 支持 JSON 输出模式
                    response_format={"type": "json_object"}, 
                    temperature=0.2,
                )
                json_result = response.choices[0].message.content
                df.at[index, 'cognitive_label'] = json_result
                success = True
                break
            except Exception as e:
                print(f"API call failed for index {index} (Attempt {attempt+1}/3): {e}")
                time.sleep(5)
        
        if not success:
            df.at[index, 'cognitive_label'] = "{}"

        if (index + 1) % 50 == 0:
            df.to_csv(OUTPUT_CSV_PATH, index=False)
            tqdm.write(f"Progress saved at sample {index+1}.")

    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print("\n" + "="*60)
    print(f"Annotation complete! Results saved to: {OUTPUT_CSV_PATH}")
    print("="*60)

if __name__ == "__main__":
    main()