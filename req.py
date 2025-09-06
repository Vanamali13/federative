import requests

data = {"features": [0]*784}  # Example MNIST input
res = requests.post("http://127.0.0.1:8000/predict", json=data)
print(res.json())
