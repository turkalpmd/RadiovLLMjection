
import json
import matplotlib.pyplot as plt
import numpy as np

def create_visualizations():
    """Analysis Results için görselleştirmeler oluştur"""
    
    # Stealth injection sonuçlarını oku
    try:
        with open('Analysis_Results/stealth_injection_semantic_analysis.json', 'r') as f:
            stealth_data = json.load(f)
    except:
        print('Stealth injection verisi bulunamadı')
        stealth_data = {}
    
    # OCR accuracy sonuçlarını oku
    try:
        with open('Analysis_Results/ocr_accuracy_semantic_analysis.json', 'r') as f:
            ocr_data = json.load(f)
    except:
        print('OCR accuracy verisi bulunamadı')
        ocr_data = {}
    
    print('🎨 GÖRSELLEŞTİRMELER OLUŞTURULUYOR...')
    
    # 1. Stealth Injection Success Rates
    if stealth_data:
        models = list(stealth_data.keys())
        success_rates = [stealth_data[m]['success_rate'] for m in models]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, success_rates, color='skyblue', alpha=0.8)
        plt.title('Stealth Injection Success Rates (Semantic Analysis)', fontsize=14, fontweight='bold')
        plt.ylabel('Success Rate (%)')
        plt.xlabel('Models')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        for bar, rate in zip(bars, success_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('Analysis_Results/stealth_injection_success_rates.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('✅ Stealth injection grafiği kaydedildi')
    
    # 2. OCR Precision/Recall/F1 Scores
    if ocr_data:
        models = list(ocr_data.keys())
        precision = [ocr_data[m]['precision'] for m in models]
        recall = [ocr_data[m]['recall'] for m in models]
        f1_scores = [ocr_data[m]['f1_score'] for m in models]
        
        x = np.arange(len(models))
        width = 0.25
        
        plt.figure(figsize=(14, 7))
        plt.bar(x - width, precision, width, label='Precision', color='lightgreen', alpha=0.8)
        plt.bar(x, recall, width, label='Recall', color='lightblue', alpha=0.8)
        plt.bar(x + width, f1_scores, width, label='F1-Score', color='lightcoral', alpha=0.8)
        
        plt.title('OCR Accuracy Metrics (Semantic Analysis)', fontsize=14, fontweight='bold')
        plt.ylabel('Score (%)')
        plt.xlabel('Models')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('Analysis_Results/ocr_accuracy_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('✅ OCR accuracy grafiği kaydedildi')
    
    # 3. Model Comparison Dashboard
    if stealth_data and ocr_data:
        common_models = set(stealth_data.keys()) & set(ocr_data.keys())
        models = list(common_models)
        
        stealth_rates = [stealth_data[m]['success_rate'] for m in models]
        f1_scores = [ocr_data[m]['f1_score'] for m in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Stealth injection
        ax1.bar(models, stealth_rates, color='orange', alpha=0.8)
        ax1.set_title('Stealth Injection Success', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Success Rate (%)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # OCR F1
        ax2.bar(models, f1_scores, color='purple', alpha=0.8)
        ax2.set_title('OCR F1-Score', fontsize=12, fontweight='bold')
        ax2.set_ylabel('F1-Score (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Model Performance Comparison Dashboard', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('Analysis_Results/model_performance_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('✅ Model comparison dashboard kaydedildi')
    
    print('\\n✅ TÜM GÖRSELLEŞTİRMELER TAMAMLANDI!')
    print('Analysis_Results klasöründe 3 görsel oluşturuldu')

if __name__ == '__main__':
    create_visualizations()
