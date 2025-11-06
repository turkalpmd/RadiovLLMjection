#!/usr/bin/env python3
"""
Process JSON result files and combine them into a CSV file.
Extracts image names (removing prefixes), aligns by cleaned image name,
and creates columns for each file's ai_result.
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd

# List of file paths (without root directory)
file_list = [
    "gpt-4.1-mini_GT_20251105_210628.json",
    "gpt-4.1-mini_injected_20251105_213417.json",
    "gpt-4.1-mini_injection_immune_20251105_234009.json",
    "gpt-4.1-mini_stealth_injected_20251105_215405.json",
    "gpt-4.1-mini_stealth_injection_immune_20251105_234928.json",
    "gpt-5_GT_20251105_222521.json",
    "gpt-5_injected_20251105_222226.json",
    "gpt-5_injection_immune_20251106_015129.json",
    "gpt-5_stealth_injected_20251105_231241.json",
    "gpt-5_stealth_injection_immune_20251106_014954.json",
    "gpt-5-nano_GT_20251105_215132.json",
    "gpt-5-nano_GT_20251105_225146.json",
    "gpt-5-nano_injected_20251105_215101.json",
    "gpt-5-nano_injection_immune_20251105_233349.json",
    "gpt-5-nano_stealth_injected_20251105_233341.json",
    "gpt-5-nano_stealth_injection_immune_20251105_234318.json",
    "anthropic_claude-sonnet-4.5_GT_20251104_222857.json",
    "anthropic_claude-sonnet-4.5_injected_20251104_223358.json",
    "anthropic_claude-sonnet-4.5_injection_immune_20251105_000635.json",
    "anthropic_claude-sonnet-4.5_stealth_injected_20251105_192127.json",
    "anthropic_claude-sonnet-4.5_stealth_injection_immune_20251106_002326.json",
    "google_gemini-2.5-flash_GT_20251104_224438.json",
    "google_gemini-2.5-flash_injected_20251105_191128.json",
    "google_gemini-2.5-flash_injection_immune_20251104_232314.json",
    "google_gemini-2.5-flash_stealth_injected_20251105_193534.json",
    "google_gemini-2.5-flash_stealth_injection_immune_20251105_191850.json",
    "microsoft_phi-4-multimodal-instruct_GT_20251104_210902.json",
    "microsoft_phi-4-multimodal-instruct_injected_20251104_211748.json",
    "microsoft_phi-4-multimodal-instruct_injection_immune_20251104_215046.json",
    "microsoft_phi-4-multimodal-instruct_stealth_injected_20251105_195101.json",
    "microsoft_phi-4-multimodal-instruct_stealth_injection_immune_20251105_193133.json",
    "nvidia_nemotron-nano-12b-v2-vl-free_GT_20251104_213847.json",
    "nvidia_nemotron-nano-12b-v2-vl-free_injected_20251104_214525.json",
    "nvidia_nemotron-nano-12b-v2-vl-free_injection_immune_20251104_221735.json",
    "nvidia_nemotron-nano-12b-v2-vl-free_stealth_injected_20251105_204407.json",
    "nvidia_nemotron-nano-12b-v2-vl-free_stealth_injection_immune_20251105_195417.json",
    "qwen_qwen2.5-vl-32b-instruct_GT_20251104_203857.json",
    "qwen_qwen2.5-vl-32b-instruct_injected_20251104_204323.json",
    "qwen_qwen3-vl-8b-thinking_GT_20251104_220732.json",
    "qwen_qwen3-vl-8b-thinking_injected_20251104_215643.json",
    "qwen_qwen3-vl-8b-thinking_injection_immune_20251105_022448.json",
    "qwen_qwen3-vl-8b-thinking_stealth_injected_20251105_212755.json",
    "qwen_qwen3-vl-8b-thinking_stealth_injection_immune_20251105_234155.json",
]

# Base directory for results
base_dir = "/home/ubuntu/RadiovLLMjection/Results"
output_dir = "/home/ubuntu/RadiovLLMjection/Results/FinalResults"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def clean_image_name(image_name):
    """Remove prefixes from image name to get base name."""
    # Remove known prefixes
    prefixes = [
        "naive_stealth_512_raw_",
        "512_raw_",
        "512_injected_",
    ]
    
    cleaned = image_name
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    
    return cleaned

def get_file_path(filename):
    """Determine the correct subdirectory for the file."""
    if filename.startswith("gpt-"):
        return os.path.join(base_dir, "OpenAI", filename)
    elif any(filename.startswith(prefix) for prefix in ["anthropic_", "google_", "microsoft_", "nvidia_", "qwen_"]):
        return os.path.join(base_dir, "OpenRouter", filename)
    else:
        # Try OpenAI first, then OpenRouter
        openai_path = os.path.join(base_dir, "OpenAI", filename)
        if os.path.exists(openai_path):
            return openai_path
        return os.path.join(base_dir, "OpenRouter", filename)

def process_json_file(filepath, filename):
    """Process a single JSON file and return dictionary mapping cleaned image names to ai_result."""
    print(f"Processing: {filename}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = {}
        
        # Handle different JSON structures
        rows = data.get('rows', [])
        if not rows:
            print(f"  Warning: No 'rows' found in {filename}")
            return results
        
        for row in rows:
            image_name = row.get('image', '')
            if not image_name:
                continue
            
            # Clean the image name
            cleaned_image = clean_image_name(image_name)
            
            # Get ai_result (prefer ai_result, fallback to ai_label)
            ai_result = row.get('ai_result')
            if ai_result is None:
                ai_result = row.get('ai_label')
            
            if ai_result is not None:
                results[cleaned_image] = ai_result
        
        print(f"  Processed {len(results)} entries")
        return results
    
    except Exception as e:
        print(f"  Error processing {filename}: {e}")
        return {}

# Process all files
all_data = {}
all_images = set()

for filename in file_list:
    filepath = get_file_path(filename)
    
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        continue
    
    # Get base filename without extension for column name
    base_name = os.path.splitext(filename)[0]
    column_name = f"{base_name}_response"
    
    # Process the file
    results = process_json_file(filepath, filename)
    
    if results:
        all_data[column_name] = results
        all_images.update(results.keys())

# Sort all image names for consistent ordering
all_images = sorted(all_images)

# Create DataFrame
data_dict = {'image': all_images}

# Add columns for each file
for column_name in sorted(all_data.keys()):
    data_dict[column_name] = [
        all_data[column_name].get(img, None) 
        for img in all_images
    ]

df = pd.DataFrame(data_dict)

# Reorder columns to put 'image' first
cols = ['image'] + [col for col in df.columns if col != 'image']
df = df[cols]

# Save to CSV
output_path = os.path.join(output_dir, "combined_results.csv")
df.to_csv(output_path, index=False)
print(f"\nOutput saved to: {output_path}")
print(f"Total images: {len(all_images)}")
print(f"Total columns: {len(df.columns)}")

