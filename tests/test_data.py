import os
import json
import pandas as pd

OUTPUT_FILE = os.path.join('..', 'outputs', 'plots', 'predictions', 'generated_questions.json')

# Sample data to append (update this with your actual data)
questions_and_distractors = [{
    'answer': 'A dragon',
    'answer_length': 8,
    'choices': ['A dragon', 'A wizard', 'A giant', 'A king'],
    'passage': 'In a distant kingdom, there lived a brave knight named Sir Cedric. Sir Cedric was famous for his many battles against the fearsome dragon that terrorized the land. One day, the dragon attacked the castle itself, and Sir Cedric knew it was time for a final showdown. Armed with his magical sword, he ventured into the dragon\'s lair, determined to end the threat once and for all.',
    'passage_length': 378,
    'question': 'Who was Sir Cedric fighting?',
    'question_length': 28,
    'question_type': 'literal'
}]

# Example Series (for demonstration)
series_example = pd.Series(['A dragon', 'A wizard', 'A giant', 'A king'])

# Function to convert non-serializable types to JSON-serializable types
def convert_to_serializable(data):
    if isinstance(data, pd.Series):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(i) for i in data]
    else:
        return data

# Read existing data
try:
    with open(OUTPUT_FILE, 'r') as f:
        existing_data = json.load(f)
        if not isinstance(existing_data, list):  # Ensure existing data is a list
            raise ValueError("The existing data is not a list.")
except FileNotFoundError:
    existing_data = []
except json.JSONDecodeError:
    print(f"Warning: {OUTPUT_FILE} contains invalid JSON. Initializing with an empty list.")
    existing_data = []
except ValueError as e:
    print(f"ValueError: {e}")
    existing_data = []
except Exception as e:
    print(f"An unexpected error occurred while reading {OUTPUT_FILE}: {e}")
    existing_data = []

# Convert any non-serializable data to a serializable format
questions_and_distractors_serializable = convert_to_serializable(questions_and_distractors)
existing_data.extend(questions_and_distractors_serializable)

# Write updated data back to the file
try:
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(existing_data, f, indent=4)
    print(f"Data successfully written to {OUTPUT_FILE}.")
except Exception as e:
    print(f"An error occurred while writing to {OUTPUT_FILE}: {e}")
