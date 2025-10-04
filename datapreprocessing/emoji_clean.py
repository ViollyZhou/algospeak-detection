import pandas as pd
import emoji
import sys

# --- Configuration ---
# Please replace 'your_data.csv' with your actual input filename
INPUT_FILE = 'D:/Pycharm/jargon_detcection/datacollection/dataset/algospeak_top_posts.csv'
# The cleaned data will be saved to this file
OUTPUT_FILE = 'emoji_data_cleaned1.csv'
# The name of the column to be cleaned
COLUMN_TO_CLEAN = 'text'


# --- Core Cleaning Function ---
def remove_emojis(text):
    """
    Removes all emojis from a string.
    """
    # Ensure the input is a string to avoid errors when processing non-text content (like NaN values)
    if not isinstance(text, str):
        return text
    return emoji.replace_emoji(text, replace='')


# --- Main Program Logic ---
def main():
    """
    Main function to load, process, and save the data.
    """
    # 1. Check for required libraries
    # (Note: This is a basic check. A more robust solution might use a requirements.txt file)
    try:
        import pandas
        import emoji
    except ImportError:
        print("Error: Required libraries are missing. Please run the following command in your terminal to install them:")
        print("pip install pandas emoji")
        sys.exit(1)  # Exit the script

    # 2. Read the data file
    try:
        print(f"Reading data from '{INPUT_FILE}'...")
        df = pd.read_csv(INPUT_FILE)
        print("File read successfully.")
    except FileNotFoundError:
        print(f"Error: File not found at '{INPUT_FILE}'.")
        print("Please make sure your data file is in the same directory as this script and the filename is correct.")
        return  # End the function

    # 3. Check if the target column exists
    if COLUMN_TO_CLEAN not in df.columns:
        print(f"Error: Column named '{COLUMN_TO_CLEAN}' not found in the file.")
        print(f"Available columns in the file are: {list(df.columns)}")
        return

    # 4. Clean emojis from the specified column
    print(f"Cleaning emojis from the '{COLUMN_TO_CLEAN}' column...")
    # We create a new column to store the cleaned text for easy comparison
    df['text_cleaned'] = df[COLUMN_TO_CLEAN].apply(remove_emojis)
    print("Cleaning complete.")

    # 5. Display a comparison of before and after cleaning (first 5 rows)
    print("\n--- Cleaning Effect Preview (First 5 Rows) ---")
    print(df[[COLUMN_TO_CLEAN, 'text_cleaned']].head())

    # 6. Save the cleaned DataFrame to a new CSV file
    try:
        print(f"\nSaving the results to '{OUTPUT_FILE}'...")
        # Using 'utf-8-sig' encoding ensures better compatibility with software like Excel
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"Success! The cleaned data has been saved to '{OUTPUT_FILE}'.")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")


# When this script is run directly, execute the main() function
if __name__ == "__main__":
    main()
