import urllib.request
import csv
import io

candidates = [
    "https://raw.githubusercontent.com/davin11/entropy-based-text-detector/main/data/train_essays.csv",
    "https://raw.githubusercontent.com/iamjr15/Ensemble-AI-Text-Detection/main/data/train_essays.csv",
    "https://raw.githubusercontent.com/Lizhecheng02/Kaggle-LLM-Detect_AI_Generated_Text/main/train_essays.csv",
    "https://raw.githubusercontent.com/biluko/LLM-Detect-AI-Generated-Text/main/data/train1.csv",
    "https://raw.githubusercontent.com/biluko/LLM-Detect-AI-Generated-Text/main/data/train_essays.csv"
]

print("Checking candidate URLs for train_essays.csv on raw.githubusercontent.com...")
for url in candidates:
    print(f"Checking: {url}")
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            status = response.status
            print(f"  Status: {status}")
            if status == 200:
                head = response.read(1024).decode('utf-8', errors='ignore')
                print("  Preview:")
                print(head[:300])
                print("-" * 50)
    except Exception as e:
        print(f"  Failed: {e}")
