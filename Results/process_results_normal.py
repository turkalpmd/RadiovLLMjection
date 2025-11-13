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
"/home/ubuntu/RadiovLLMjection/Results/OpenRouter/normal/anthropic_claude-sonnet-4.5_GT_20251113_085529.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenRouter/normal/anthropic_claude-sonnet-4.5_injected_20251113_093830.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenRouter/normal/anthropic_claude-sonnet-4.5_injection_immune_20251113_153431.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenRouter/normal/anthropic_claude-sonnet-4.5_stealth_injected_20251113_111032.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenRouter/normal/anthropic_claude-sonnet-4.5_stealth_injection_immune_20251113_150509.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenRouter/normal/google_gemini-2.5-flash_GT_20251113_080901.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenRouter/normal/google_gemini-2.5-flash_injected_20251113_085926.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenRouter/normal/google_gemini-2.5-flash_injection_immune_20251113_093310.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenRouter/normal/google_gemini-2.5-flash_stealth_injected_20251113_090956.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenRouter/normal/google_gemini-2.5-flash_stealth_injection_immune_20251113_092134.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenRouter/normal/microsoft_phi-4-multimodal-instruct_GT_20251113_081226.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenRouter/normal/microsoft_phi-4-multimodal-instruct_injected_20251113_090230.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenRouter/normal/microsoft_phi-4-multimodal-instruct_injection_immune_20251113_095309.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenRouter/normal/microsoft_phi-4-multimodal-instruct_stealth_injected_20251113_091932.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenRouter/normal/microsoft_phi-4-multimodal-instruct_stealth_injection_immune_20251113_092448.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenRouter/normal/nvidia_nemotron-nano-12b-v2-vl-free_GT_20251113_082956.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenRouter/normal/nvidia_nemotron-nano-12b-v2-vl-free_injected_20251113_092313.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenRouter/normal/nvidia_nemotron-nano-12b-v2-vl-free_injection_immune_20251113_101600.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenRouter/normal/nvidia_nemotron-nano-12b-v2-vl-free_stealth_injected_20251113_101439.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenRouter/normal/nvidia_nemotron-nano-12b-v2-vl-free_stealth_injection_immune_20251113_094905.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenRouter/normal/qwen_qwen3-vl-8b-thinking_GT_20251113_084846.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenRouter/normal/qwen_qwen3-vl-8b-thinking_injected_20251113_093150.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenRouter/normal/qwen_qwen3-vl-8b-thinking_injection_immune_20251113_142807.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenRouter/normal/qwen_qwen3-vl-8b-thinking_stealth_injected_20251113_104816.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenRouter/normal/qwen_qwen3-vl-8b-thinking_stealth_injection_immune_20251113_142213.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenAI/normal/gpt-4.1-mini_GT_20251113_100033.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenAI/normal/gpt-4.1-mini_injected_20251113_101453.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenAI/normal/gpt-4.1-mini_injection_immune_20251113_111357.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenAI/normal/gpt-4.1-mini_stealth_injected_20251113_112822.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenAI/normal/gpt-4.1-mini_stealth_injection_immune_20251113_114016.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenAI/normal/gpt-5_GT_20251113_103849.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenAI/normal/gpt-5_injected_20251113_103805.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenAI/normal/gpt-5_injection_immune_20251113_135306.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenAI/normal/gpt-5_stealth_injected_20251113_130128.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenAI/normal/gpt-5_stealth_injection_immune_20251113_142116.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenAI/normal/gpt-5-nano_GT_20251113_095353.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenAI/normal/gpt-5-nano_injected_20251113_100707.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenAI/normal/gpt-5-nano_injection_immune_20251113_110300.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenAI/normal/gpt-5-nano_stealth_injected_20251113_110941.json",
"/home/ubuntu/RadiovLLMjection/Results/OpenAI/normal/gpt-5-nano_stealth_injection_immune_20251113_112228.json",
]

# Base directory for results
base_dir = "/home/ubuntu/RadiovLLMjection/Results"
output_dir = "/home/ubuntu/RadiovLLMjection/Results/FinalResults_for_Normals"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def clean_image_name(image_name):
    """Remove prefixes from image name to get base name."""
    # Remove file extension if present
    cleaned = image_name
    if '.' in cleaned:
        cleaned = os.path.splitext(cleaned)[0]
    
    # Pattern: extract number from parentheses for specific patterns
    # Priority: more specific patterns first
    patterns = [
        r'injected_raw_normal\s*\((\d+)\)',  # injected_raw_normal (123) -> 123
        r'stealth_normal_normal\s*\((\d+)\)',  # stealth_normal_normal (123) -> 123
        r'normal\s*\((\d+)\)',  # normal (123) -> 123
    ]
    
    for pattern in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            cleaned = match.group(1)
            return cleaned
    
    # Fallback: remove known prefixes if pattern matching didn't work
    prefixes = [
        "naive_stealth_512_raw_",
        "512_raw_",
        "512_injected_",
    ]
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

for filepath in file_list:
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        continue
    
    # Get base filename without extension for column name
    filename = os.path.basename(filepath)
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

