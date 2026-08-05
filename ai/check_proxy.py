import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

proxy_url = os.getenv("OPENAI_PROXY")
api_key = os.getenv("OPENAI_API_KEY")

def test_proxy_connection(proxy_url, api_key):
    print(f"proxy_url is NOT none" if proxy_url else "proxy url is none")
    print(f"api_key is NOT none" if api_key else "api key is none")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Test 1: Checking API availability
    try:
        response = httpx.get(
            "https://api.openai.com/v1/models",
            headers=headers,
            proxy=proxy_url,
            timeout=30,
        )

        if response.status_code == 200:
            print("✓ Proxy works with OpenAI API")
            print(f"Models available: {len(response.json()['data'])}")
        else:
            print(f"✗ Error: HTTP {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"✗ Connection error: {e}")
        return False

def test_proxy_speed(proxy_url, api_key) -> bool:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    start_time = time.time()

    test_data = {
        "model": "gpt-5.2",
        "messages": [
            {"role": "user", "content": "Hi"},
        ],
        "max_completion_tokens": 10,
    }

    try:
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=test_data,
            proxy=proxy_url,
            timeout=30,
        )

        elapsed = time.time() - start_time

        if response.status_code == 200:
            print(f"✓ Speed test OK, response time: {elapsed:.2f} sec")
            return True

        print(f"✗ Speed test failed: HTTP {response.status_code}")
        print(response.text)
        return False

    except Exception as e:
        print(f"✗ Speed test exception: {e}")
        return False



if __name__ == '__main__':
    test_proxy_connection(proxy_url, api_key)
    test_proxy_speed(proxy_url, api_key)
