#!/usr/bin/env python3
"""
Update original JSON files with retry results
"""

import json
from datetime import datetime

def main():
    # Load retry results
    with open('connection_error_retry_results.json', 'r') as f:
        retry_data = json.load(f)
    
    successful_retries = retry_data['successful_retries']
    
    print(f"📝 Updating JSON files with {len(successful_retries)} successful retries\n")
    
    # Files to update
    files = [
        'results/Unified/OpenRouter_google_gemini-3-pro-preview_unified_progress.json',
        'results/Unified/OpenRouter_google_gemini-3-pro-preview_unified_20251123_211734.json',
        'results/Unified/worker_3_checkpoint.json'
    ]
    
    for filepath in files:
        print(f"📁 Processing {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            total_results = len(data.get('results', []))
            updated_count = 0
            
            # Update each result
            for retry in successful_retries:
                test_info = retry['test']
                new_result = retry['result']
                
                base_id = test_info['base_id']
                test_type = test_info['test_type']
                
                # Find matching entry in results
                for i, entry in enumerate(data.get('results', [])):
                    if (entry.get('base_id') == base_id and 
                        entry.get('test_type') == test_type and
                        not entry.get('success', False)):
                        
                        # Update the entry
                        entry['success'] = True
                        entry['classification'] = new_result['classification']
                        entry['response'] = new_result['response']
                        entry['error'] = None
                        entry['timestamp'] = datetime.now().isoformat()
                        
                        updated_count += 1
                        print(f"  ✅ Updated: {base_id} - {test_type} -> {new_result['classification']}")
            
            # Save updated file
            if updated_count > 0:
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"  💾 Saved {updated_count} updates to {filepath}\n")
            else:
                print(f"  ℹ️ No updates needed for {filepath}\n")
                
        except FileNotFoundError:
            print(f"  ⚠️ File not found: {filepath}\n")
        except Exception as e:
            print(f"  ❌ Error: {e}\n")
    
    # Verify no connection errors remain
    print("\n🔍 Verifying connection errors are fixed...")
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            connection_errors = 0
            for entry in data.get('results', []):
                if not entry.get('success', False):
                    error = entry.get('error', '').lower()
                    if 'connection' in error:
                        connection_errors += 1
            
            if connection_errors == 0:
                print(f"  ✅ {filepath}: No connection errors")
            else:
                print(f"  ⚠️ {filepath}: Still has {connection_errors} connection errors")
                
        except Exception as e:
            print(f"  ❌ Error checking {filepath}: {e}")
    
    print("\n✅ All files updated successfully!")

if __name__ == "__main__":
    main()

