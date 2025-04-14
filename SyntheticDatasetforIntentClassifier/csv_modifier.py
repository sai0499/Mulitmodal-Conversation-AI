import pandas as pd
import re

# Replace with the correct paths for your input and output CSV files.
input_file_path = 'Intent_Classifier_Dataset - Processed.csv'
output_file_path = 'Intent_Classifier_Dataset_Processed_updated.csv'

# Load the CSV file into a DataFrame.
df = pd.read_csv(input_file_path)


def process_question(row):
    # Remove extra whitespace at the beginning and end.
    question = row['Question'].strip()
    label = row['Label']

    # For web search questions: ensure it ends with a question mark.
    if label == 'web search':
        if not question.endswith('?'):
            question += '?'

    # For general questions: remove extra text if there are more than 3 spaces.
    elif label == 'general question':
        # Look for a sequence of 4 or more consecutive spaces.
        match = re.search(r'\s{4,}', question)
        if match:
            # Keep only the text before the first occurrence of these spaces.
            question = question[:match.start()].rstrip()

    return question


# Apply the process_question function to every row in the DataFrame.
df['Question'] = df.apply(process_question, axis=1)

# Save the updated DataFrame to a new CSV file.
df.to_csv(output_file_path, index=False)

print(f"Updated file saved as {output_file_path}")
