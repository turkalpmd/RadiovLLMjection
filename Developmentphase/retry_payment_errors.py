#!/usr/bin/env python3
"""Retry payment error tests"""

import json
import os
import pandas as pd
from dotenv import load_dotenv
from unified_radiology_tester import UnifiedRadiologyTester
from datetime import datetime

load_dotenv()

def main():
    # Load payment errors
    with open('payment_errors_to_retry.json', 'r') as f:
        payment_errors = json.load(f)
    
    print(f"🔄 Retrying {len(payment_errors)} payment error tests\n")
    
    # Initialize tester
    tester = UnifiedRadiologyTester(
        dataset_csv='dataset.csv',
        output_dir='results/PaymentErrorRetries'
    )
    
    # Load dataset
    dataset = pd.read_csv('dataset.csv')
    
    # Model config
    model_config = {
        'provider': 'OpenRouter',
        'model': 'google/gemini-3-pro-preview'
    }
    
    successful_retries = []
    still_failing = []
    
    for i, test in enumerate(payment_errors, 1):
        base_id = test['base_id']
        test_type = test['test_type']
        
        print(f"[{i}/{len(payment_errors)}] {base_id} - {test_type}")
        
        try:
            # Find in dataset
            matching_rows = dataset[dataset['base_id'] == base_id]
            
            if matching_rows.empty:
                print(f"  ⚠️ Not found in dataset")
                still_failing.append({'test': test, 'error': 'Not in dataset'})
                continue
            
            image_row = matching_rows.iloc[0]
            
            # Retry the test
            result = tester.run_single_test(
                model_config=model_config,
                test_type=test_type,
                image_row=image_row
            )
            
            if result.success:
                print(f"  ✅ Success! Classification: {result.classification}")
                successful_retries.append({
                    'test': test,
                    'result': {
                        'base_id': base_id,
                        'test_type': test_type,
                        'classification': result.classification,
                        'response': result.response,
                        'ground_truth': test['ground_truth'],
                        'success': True,
                        'error': None,
                        'timestamp': datetime.now().isoformat()
                    }
                })
            else:
                error_msg = result.error or 'Unknown error'
                print(f"  ❌ Failed: {error_msg[:80]}")
                still_failing.append({'test': test, 'error': error_msg})
                
        except Exception as e:
            print(f"  ❌ Exception: {str(e)[:80]}")
            still_failing.append({'test': test, 'error': str(e)})
    
    # Save results
    retry_summary = {
        'total': len(payment_errors),
        'successful': len(successful_retries),
        'failed': len(still_failing),
        'successful_retries': successful_retries,
        'still_failing': still_failing
    }
    
    with open('payment_error_retry_results.json', 'w') as f:
        json.dump(retry_summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ Successful: {len(successful_retries)}/{len(payment_errors)}")
    print(f"❌ Failed: {len(still_failing)}/{len(payment_errors)}")
    print(f"\nResults saved to payment_error_retry_results.json")
    
    # Update original file if successful
    if successful_retries:
        print(f"\n🔧 Updating unified_progress.json...")
        update_json_file(successful_retries)
    
    return 0 if len(still_failing) == 0 else 1


def update_json_file(successful_retries):
    """Update the unified_progress.json with successful results"""
    
    filepath = 'results/Unified/OpenRouter_google_gemini-3-pro-preview_unified_progress.json'
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        updated_count = 0
        
        for retry in successful_retries:
            test_info = retry['test']
            new_result = retry['result']
            
            base_id = test_info['base_id']
            test_type = test_info['test_type']
            
            # Find and update
            for entry in data.get('results', []):
                if (entry.get('base_id') == base_id and 
                    entry.get('test_type') == test_type and
                    not entry.get('success', False)):
                    
                    entry['success'] = True
                    entry['classification'] = new_result['classification']
                    entry['response'] = new_result['response']
                    entry['error'] = None
                    entry['timestamp'] = new_result['timestamp']
                    
                    updated_count += 1
                    print(f"  ✅ Updated: {base_id} - {test_type} -> {new_result['classification']}")
        
        # Save
        if updated_count > 0:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"  💾 Saved {updated_count} updates")
        
    except Exception as e:
        print(f"  ❌ Error updating file: {e}")


if __name__ == "__main__":
    exit(main())

