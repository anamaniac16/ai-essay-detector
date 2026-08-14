import os
import urllib.request

LOCAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "gpt2_local")
os.makedirs(LOCAL_DIR, exist_ok=True)

files = {
    "config.json": "https://huggingface.co/gpt2/resolve/main/config.json",
    "vocab.json": "https://huggingface.co/gpt2/resolve/main/vocab.json",
    "merges.txt": "https://huggingface.co/gpt2/resolve/main/merges.txt",
    "tokenizer_config.json": "https://huggingface.co/gpt2/resolve/main/tokenizer_config.json",
    "model.safetensors": "https://huggingface.co/gpt2/resolve/main/model.safetensors"
}

print("Downloading GPT-2 files via plain HTTPS GET to avoid HEAD connection resets...")
for name, url in files.items():
    dest = os.path.join(LOCAL_DIR, name)
    if os.path.exists(dest):
        print(f"  {name} already exists, skipping.")
        continue
        
    print(f"  Downloading {name} from {url} ...")
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=120) as response, open(dest, "wb") as out_file:
            # Buffer reading to handle large files nicely
            chunk_size = 1024 * 1024
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
        print(f"  Successfully saved to {dest}")
    except Exception as e:
        print(f"  Failed to download {name}: {e}")
        if os.path.exists(dest):
            os.remove(dest)
