import requests
import json

url = "http://localhost:8088/chat/stream"
query = """ summarize the following customer scores
Customer Scores:
drTurnOverAbs: 34
currentBalanceAmountScore: 40
propDrStocrScore: 33
mobileTotalScore: 45
pastDueAmountScore: 52
maxArrearsScore: 98
currentArrearsScore: 58
nonPerformingScore: 60
avgLnMPaymentsScore: 105
customerTenureScore: 44
ageScore: 46
savingAcctDepositCountScore: 48
employerStrengthScore: 10
totalConsumerScore: 585
"""

payload = {"query": query, "user_id": "default"}

print(f"Testing Streaming Endpoint: {url}")
try:
    with requests.post(url, json=payload, stream=True) as response:
        if response.status_code == 200:
            print("Response Stream:")
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    print(chunk, end='', flush=True)
            print("\n\nStream finished.")
        else:
            print(f"Status Code: {response.status_code}")
            print(response.text)
except Exception as e:
    print(f"Error: {e}")
