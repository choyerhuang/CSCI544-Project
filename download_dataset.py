import json
import requests
import random


math_500_path = "math_500.json"
sample_100_path = "sample_100.json"


# %%
# Download the dataset from Hugging Face

merged_rows = []

for offset in range(0, 401, 100):
    paginated_url = f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4%2FMATH-500&config=default&split=test&offset={offset}&length=100"
    response = requests.get(paginated_url)
    data = response.json()
    merged_rows.extend(data["rows"])
    
data["rows"] = merged_rows
del data["num_rows_per_page"]
del data["partial"]

with open(math_500_path, "w") as f:
    json.dump(data, f, indent=2)


# %%
# Sample 100 unique_ids from the dataset

# Group rows by their level
rows_by_level = {}
for row in data["rows"]:
    level = row["row"]["level"]
    if level not in rows_by_level:
        rows_by_level[level] = []
    rows_by_level[level].append(row)

# Randomly sample 20 objects for each level
sampled_rows = {}
for level, rows in rows_by_level.items():
    sampled_rows[level] = random.sample(rows, min(20, len(rows)))

# Flatten the sampled rows into a single list
sampled_rows_flat = [row for rows in sampled_rows.values() for row in rows]

# Sort the sampled rows by level and then by row_idx
sampled_rows_flat.sort(key=lambda x: (x["row"]["level"], x["row_idx"]))

# print(f"Total sampled rows: {len(sampled_rows_flat)}")
# print(json.dumps(sampled_rows_flat, indent=2))

# Extract unique_id from the sampled rows
unique_ids = [row["row"]["unique_id"] for row in sampled_rows_flat]

# Print the list of unique_ids
# print(f"Extracted unique_ids:")
# print(json.dumps(unique_ids, indent=2))

with open(sample_100_path, "w") as f:
    json.dump(unique_ids, f, indent=2)
