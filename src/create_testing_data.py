import pandas as pd

# Read the main dataset
data = pd.read_csv("Data/raw_reviews.csv")

# Take 100 random rows
testing_data = data.sample(n=100, random_state=42)

# Save the new dataset
testing_data.to_csv("Data/testing_data.csv", index=False)

print("Testing dataset created successfully")