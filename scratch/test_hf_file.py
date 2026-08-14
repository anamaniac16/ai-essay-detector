import urllib.request

urls = [
    "https://huggingface.co/gpt2/resolve/main/config.json",
    "https://huggingface.co/gpt2/resolve/main/vocab.json",
    "https://huggingface.co/gpt2/resolve/main/merges.txt",
    "https://huggingface.co/gpt2/resolve/main/tokenizer_config.json"
]

for url in urls:
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            print(f"{url.split('/')[-1]} -> Status: {r.status}, Size: {len(r.read())}")
    except Exception as e:
        print(f"{url.split('/')[-1]} -> Failed: {e}")
