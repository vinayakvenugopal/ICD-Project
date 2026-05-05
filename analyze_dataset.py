import pandas as pd
import os

csv_path = r'c:\Users\acer\Downloads\Telegram Desktop\ICD (2)\ICD\ml\full_raw_codes.csv'
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f'Total rows: {len(df)}')
    code_col = 'ICDCode' if 'ICDCode' in df.columns else ('code' if 'code' in df.columns else None)
    if code_col:
        print(f'Unique codes in {code_col}: {df[code_col].nunique()}')
        print('Top 20 codes distribution:')
        print(df[code_col].value_counts().head(20))
    else:
        print(f"Columns: {df.columns}")
else:
    print(f"File not found: {csv_path}")
