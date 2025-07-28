import json

test_payload = {"test": "value", "another": 123}

with open("payload.json", "w", encoding="utf-8") as f:
    json.dump(test_payload, f, indent=2)

print("File written successfully.")