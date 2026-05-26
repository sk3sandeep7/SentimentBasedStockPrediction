import pandas as pd
import re
import html
import emoji

def clean_tweet(text):
    """
    Applies the 5 baseline cleaning operations to a single string of text.
    """
    # Handle empty or missing values
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # 1. Decode HTML Entities (e.g., &amp; -> &)
    text = html.unescape(text)

    # 2. Strip URLs & Image Links
    text = re.sub(r'http[s]?://\S+|www\.\S+|pic\.twitter\.com/\S+', '', text)

    # 3. Remove User Mentions
    text = re.sub(r'@\w+', '', text)

    # 4. Standardize Tickers (Assuming TSLA based on the global clean rules)
    # If you need to add more hardcoded tickers like GME, you can expand this regex
    text = re.sub(r'\b(tesla|tsla)\b|#tsla|\$tsla', '$TSLA', text, flags=re.IGNORECASE)

    # 5. Convert Emojis to Text (e.g., 🚀 -> rocket)
    text = emoji.demojize(text)
    text = re.sub(r':([a-zA-Z0-9_]+):', lambda m: ' ' + m.group(1).replace('_', ' ') + ' ', text)

    # 6. Cleanup Extraneous Whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def process_dataset(input_csv, output_csv):
    print(f"Reading data from {input_csv}...")
    
    try:
        # Read the first 500 lines
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Could not find '{input_csv}'.")
        return

    # Find the exact name of the tweet column (handling potential capitalization differences like 'Tweet' or 'TWEET')
    target_col = None
    for col in df.columns:
        if str(col).strip().lower() == 'tweet':
            target_col = col
            break
            
    if not target_col:
        print(f"Error: Could not find a column named 'tweet'. Available columns are: {list(df.columns)}")
        return

    print("Iterating through rows and cleaning tweets...")
    
    # The For-Loop Implementation
    for i, text in enumerate(df[target_col]):
        # 1. Get the raw text from the tweet column
        #raw_text = df.at[index, target_col]
        
        # 2. Pass ONLY the text to the cleaning function
        cleaned_text = clean_tweet(text)
        
        # 3. Update the tweet column in the dataframe
        df.at[i, target_col] = cleaned_text

    # Save the dataframe. Because we only updated the 'tweet' column, 
    # all other columns (date, ticker, company name, etc.) remain exactly as they were.
    print(f"Saving updated data to {output_csv}...")
    df.to_csv(output_csv, index=False)
    print("Done!")

# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    # Replace 'YOUR_FILE_NAME.csv' with your actual file
    INPUT_FILE = '/Users/blaze/Repos/Sentiment Analyzer/KaggleDataSets/stock_tweets.csv' 
    OUTPUT_FILE = '/Users/blaze/Repos/Sentiment Analyzer/KaggleDataSets/cleaned_tweets.csv'
    
    process_dataset(INPUT_FILE, OUTPUT_FILE)