import requests

url = "http://localhost:8088/chat"
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

payload = {
    "query": query,
    "user_id": "default"
}

try:
    print("Testing Credit Scoring Summary...")
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        data = response.json()
        print("\n--- Model Response ---\n")
        print(data["response"])
        if "metrics" in data and data["metrics"]:
            print("\n--- Performance Metrics ---")
            for metric, val in data["metrics"].items():
                print(f"{metric:20}: {val}s")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Connection failed: {e}")
