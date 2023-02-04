import requests
import json

TEST_FILE = './data/requests.json'

if __name__ == "__main__":
    with open(TEST_FILE, 'r') as f:
        for l in f:
            r = json.loads(l)
            requests.post(url='http://127.0.0.1:8000/predict', json=r)