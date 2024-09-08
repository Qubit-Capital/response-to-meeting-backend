import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve MongoDB URI and CSV file path from environment variables
mongodb_uri = os.getenv('MONGODB_URI')
csv_file_path = os.getenv('CSV_FILE_PATH')

# MongoDB connection
client = MongoClient(mongodb_uri)
db = client['response-to-meeting']
collection = db['emails']

# Load CSV data
df = pd.read_csv(csv_file_path)

# Select only the desired fields
selected_fields = [
    'from_email', 'subject', 'to_email', 'to_name', 
    'event_timestamp', 'campaign_name', 'campaign_id', 
    'sent_message_text', 'reply_message_text', 'time_replied'
]
df_selected = df[selected_fields]

# Add the new field `status` with the initial value 'standard'
df_selected['status'] = 'standard'

# Convert DataFrame to a list of dictionaries
data_to_insert = df_selected.to_dict(orient='records')

# Insert data into MongoDB
collection.insert_many(data_to_insert)

print(f"Successfully inserted {len(data_to_insert)} documents into the collection.")


