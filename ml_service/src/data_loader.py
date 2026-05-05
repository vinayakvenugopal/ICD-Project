import pandas as pd
import os

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None

    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        
        # Detect if it's the raw file (no header, multiple columns) or processed file
        try:
            # Try reading first few lines to detect format
            with open(self.data_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                
            if first_line.startswith('code,'):
                # Processed format
                self.df = pd.read_csv(self.data_path)
            else:
                # Raw format: A00,0,A000,"Desc"...
                # We need columns 2 (code) and 4 (long description)
                raw_df = pd.read_csv(self.data_path, header=None, quoting=1)
                self.df = pd.DataFrame({
                    'code': raw_df[2],
                    'description': raw_df[4]
                })
                # Format codes (A000 -> A00.0)
                self.df['code'] = self.df['code'].apply(lambda x: x[:3] + "." + x[3:] if len(str(x)) > 3 else str(x))
                
        except Exception as e:
            print(f"Error parsing dataset: {e}. Falling back to standard load.")
            self.df = pd.read_csv(self.data_path)
            
        return self.df

    def get_descriptions(self):
        if self.df is None:
            self.load_data()
        return self.df['description'].tolist()

    def get_code_map(self):
        if self.df is None:
            self.load_data()
        return dict(zip(self.df['description'], self.df['code']))
