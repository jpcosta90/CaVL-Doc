
import pandas as pd
import sys

file_path = '/home/joaopaulo/Projects/CaVL-Doc/docs/paper/references/introduction-and-state-of-art/relevant_articles_filtered.xlsx'

try:
    # Read the excel file
    df = pd.read_excel(file_path)
    
    # Print columns to confirm
    print("Columns found:", df.columns.tolist())
    
    # Check if 'Foco' exists
    if 'Foco' in df.columns:
        # Filter rows where Foco is not null/empty
        focused_df = df[df['Foco'].notna() & (df['Foco'] != '')]
        
        print(f"\nFound {len(focused_df)} articles with notes:")
        print("-" * 50)
        
        for index, row in focused_df.iterrows():
            title = row.get('Title', 'No Title')
            foco = row.get('Foco', '')
            print(f"TITLE: {title}")
            print(f"NOTE (FOCO): {foco}")
            print("-" * 50)
            
    else:
        print("Column 'Foco' not found in the Excel file.")
        print("First few rows:")
        print(df.head())

except Exception as e:
    print(f"Error reading Excel file: {e}")
