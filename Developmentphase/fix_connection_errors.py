#!/usr/bin/env python3
"""
Fix connection errors by retrying tests and updating original JSON files
"""

import json
import os
import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from unified_radiology_tester import UnifiedRadiologyTester
from collections import defaultdict

# Load environment
load_dotenv()

def main():
    # Check API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found")
        return 1
    
    print(f"✅ API Key loaded")
    
    # Load failed tests
    with open('connection_errors_to_retry.json') as f:
        failed_tests = json.load(f)
    
    print(f"\n📊 Retrying {len(failed_tests)} connection error tests\n")
    
    # Initialize tester
    tester = UnifiedRadiologyTester(
        dataset_csv='dataset.csv',
        output_dir='results/ConnectionErrorRetries'
    )
    
    # Load full dataset
    dataset = pd.read_csv('dataset.csv')
    
    # Model configuration
    model_config = {
        'provider': 'OpenRouter',
        'model': 'google/gemini-3-pro-preview'
    }
    
    # Group failed tests by file
    tests_by_file = defaultdict(list)
    for test in failed_tests:
        # Determine which file this test belongs to
        base_id = test['base_id']
        if int(base_id.split('_')[1]) < 225:
            file_key = 'worker_3'
        else:
            file_key = 'unified'
        tests_by_file[file_key].append(test)
    
    # Track results
    successful_retries = []
    still_failing = []
    
    # Retry each test
    for i, test in enumerate(failed_tests, 1):
        base_id = test['base_id']
        test_type = test['test_type']
        image_file = Path(test['image']).name
        
        print(f"[{i}/{len(failed_tests)}] {image_file} - {test_type}")
        
        try:
            # Find the row in dataset
            matching_rows = dataset[dataset['base_id'] == base_id]
            
            if matching_rows.empty:
                print(f"  ⚠️ {base_id} not found in dataset")
                still_failing.append({
                    'test': test,
                    'error': 'Not found in dataset'
                })
                continue
            
            image_row = matching_rows.iloc[0]
            
            # Run the test
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
                        'error': None
                    }
                })
            else:
                error_msg = result.error if result.error else 'Unknown error'
                print(f"  ❌ Failed: {error_msg[:60]}")
                still_failing.append({
                    'test': test,
                    'error': error_msg
                })
                
        except Exception as e:
            print(f"  ❌ Exception: {str(e)[:60]}")
            still_failing.append({
                'test': test,
                'error': str(e)
            })
    
    # Save retry results
    retry_summary = {
        'total_retried': len(failed_tests),
        'successful': len(successful_retries),
        'still_failing': len(still_failing),
        'success_rate': len(successful_retries) / len(failed_tests) * 100 if failed_tests else 0,
        'successful_retries': successful_retries,
        'still_failing': still_failing
    }
    
    with open('connection_error_retry_results.json', 'w') as f:
        json.dump(retry_summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"📈 RETRY SUMMARY")
    print(f"{'='*70}")
    print(f"Total retried: {retry_summary['total_retried']}")
    print(f"✅ Successful: {retry_summary['successful']} ({retry_summary['success_rate']:.1f}%)")
    print(f"❌ Still failing: {retry_summary['still_failing']}")
    print(f"\nResults saved to: connection_error_retry_results.json")
    
    if successful_retries:
        print(f"\n🔧 Now updating original JSON files...")
        update_original_files(successful_retries)
    
    return 0 if len(still_failing) == 0 else 1


def update_original_files(successful_retries):
    """Update the original JSON files with successful retry results"""
    
    # Group by file
    files_to_update = {
        'results/Unified/OpenRouter_google_gemini-3-pro-preview_unified_progress.json': [],
        'results/Unified/OpenRouter_google_gemini-3-pro-preview_unified_20251123_211734.json': [],
        'results/Unified/worker_3_checkpoint.json': []
    }
    
    for retry in successful_retries:
        test = retry['test']
        result = retry['result']
        base_id = test['base_id']
        
        # Determine which files to update
        if int(base_id.split('_')[1]) < 225:
            files_to_update['results/Unified/worker_3_checkpoint.json'].append((test, result))
        else:
            files_to_update['results/Unified/OpenRouter_google_gemini-3-pro-preview_unified_progress.json'].append((test, result))
            files_to_update['results/Unified/OpenRouter_google_gemini-3-pro-preview_unified_20251123_211734.json'].append((test, result))
    
    # Update each file
    for filepath, updates in files_to_update.items():
        if not updates:
            continue
            
        print(f"\n📝 Updating {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            updated_count = 0
            for test, new_result in updates:
                base_id = test['base_id']
                test_type = test['test_type']
                
                # Find and update the result
                for i, result_entry in enumerate(data.get('results', [])):
                    if result_entry.get('base_id') == base_id:
                        # Update the specific test type field
                        test_key = f"{test_type}_result"
                        if test_key in result_entry:
                            result_entry[test_key]['success'] = True
                            result_entry[test_key]['classification'] = new_result['classification']
                            result_entry[test_key]['response'] = new_result['response']
                            result_entry[test_key]['error'] = None
                            updated_count += 1
                            print(f"  ✅ Updated {base_id} - {test_type}")
            
            # Save updated file
            if updated_count > 0:
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"  💾 Saved {updated_count} updates")
            
        except Exception as e:
            print(f"  ❌ Error updating {filepath}: {e}")


if __name__ == "__main__":
    sys.exit(main())

