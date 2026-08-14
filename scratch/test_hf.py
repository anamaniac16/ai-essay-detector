import urllib.request

try:
    url = "https://huggingface.co"
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(req, timeout=5) as response:
        print(f"HuggingFace.co Status: {response.status}")
except Exception as e:
    print(f"HuggingFace.co Connection Failed: {e}")
