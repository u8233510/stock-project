import requests
import json

# ====== 改成你自己的 ======
API_KEY = "AQ.Ab8RN6Ij4uS6Qc0He7s5ubfp7hr4FDCtM6n9NrlIKuWICaf62A"
CX = "8503770eaafd3464d"
QUERY = "台積電"


# ==========================

url = "https://www.googleapis.com/customsearch/v1"

params = {
    "key": API_KEY,
    "cx": CX,
    "q": QUERY,
}

print("Sending request...")
resp = requests.get(url, params=params)

print("Status Code:", resp.status_code)
print()

data = resp.json()

# 如果出錯，直接把完整訊息印出來
if "error" in data:
    print("❌ ERROR:")
    print(json.dumps(data, indent=2, ensure_ascii=False))
else:
    print("✅ SUCCESS\n")

    for i, item in enumerate(data.get("items", []), start=1):
        print(f"{i}. {item['title']}")
        print(item['link'])
        print("-" * 60)
