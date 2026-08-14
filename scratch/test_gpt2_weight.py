import urllib.request

urls = [
    "https://huggingface.co/gpt2/resolve/main/pytorch_model.bin",
    "https://huggingface.co/gpt2/resolve/main/model.safetensors"
]

for url in urls:
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        # Just read a tiny bit of headers or check response
        with urllib.request.urlopen(req, timeout=10) as r:
            print(f"{url.split('/')[-1]} -> Status: {r.status}, Content-Length: {r.headers.get('Content-Length')}")
    except Exception as e:
        print(f"{url.split('/')[-1]} -> Failed: {e}")
