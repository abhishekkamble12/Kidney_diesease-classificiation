import os
import sys

def test_distilbert():
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    from pathlib import Path
    
    distilbert_path = Path('artifacts/model_trainer/model_DistilBERT')
    if distilbert_path.exists():
        model_source = str(distilbert_path)
    else:
        model_source = 'distilbert-base-uncased-finetuned-sst-2-english'
        
    print(f'Source: {model_source}')
    
    try:
        from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
        print('Transformers imported successfully')
        
        tokenizer = AutoTokenizer.from_pretrained(model_source)
        print('Tokenizer loaded successfully')
        
        model = TFAutoModelForSequenceClassification.from_pretrained(model_source)
        print('Model loaded successfully')
        
    except Exception as e:
        print(f'Error loading DistilBERT: {e}')
        sys.exit(1)

if __name__ == "__main__":
    test_distilbert()
