import json
import pandas as pd

# Load the JSON data from file.
with open('WebQSP.test.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Create a DataFrame from the "Questions" list.
df = pd.DataFrame(data['Questions'])

# Extract the "ProcessedQuestion" column.
processed_questions_df = df[['ProcessedQuestion']]

# Write the processed questions to a CSV file.
processed_questions_df.to_csv('processed_questions2.csv', index=False)

print("CSV file 'processed_questions.csv' has been created with the processed questions.")
