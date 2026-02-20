import os
import pandas as pd
import argparse

def concatenate_files_in_directory(directory_path, output_file=None):
    """
    Concatenate files in a directory based on alphabetical order.
    Assumes files are CSV format.
    """
    # Get all files in directory and sort them
    files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    files.sort()
    
    if not files:
        print("No CSV files found in directory")
        return
    
    # Read and concatenate files
    dataframes = []
    total_rows = 0
    for i, file in enumerate(files):
        file_path = os.path.join(directory_path, file)
        # Skip header for all files except the first one
        if i == 0:
            df = pd.read_csv(file_path)
            print(f"First file columns: {list(df.columns)}")
        else:
            df = pd.read_csv(file_path, skiprows=1)
        
        # Check column differences
        if i > 0:
            first_df_cols = set(dataframes[0].columns)
            current_df_cols = set(df.columns)
            missing_in_current = first_df_cols - current_df_cols
            extra_in_current = current_df_cols - first_df_cols
            if missing_in_current or extra_in_current:
                print(f"⚠️  {file} has different columns:")
                if missing_in_current:
                    print(f"   Missing: {missing_in_current}")
                if extra_in_current:
                    print(f"   Extra: {extra_in_current}")
        
        # Check for NaNs in this file
        if df.isnull().any().any():
            nan_columns = df.columns[df.isnull().any()].tolist()
            print(f"⚠️  {file} contains NaNs in columns: {nan_columns}")
        
        dataframes.append(df)
        print(f"Added {file} starting at row {total_rows} with {len(df)} rows")
        total_rows += len(df)

    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Check for NaNs in combined dataset
    if combined_df.isnull().any().any():
        nan_columns = combined_df.columns[combined_df.isnull().any()].tolist()
        print(f"\n⚠️  Combined dataset contains NaNs in columns: {nan_columns}")
        print(f"Total NaN count: {combined_df.isnull().sum().sum()}")
    return combined_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate CSV files in a directory")
    parser.add_argument("--dir", help="Path to directory containing CSV files")
    parser.add_argument("--o", "--output", help="Output file path (optional)")
    
    args = parser.parse_args()
    print(args)
    
    concatenate_files_in_directory(args.dir, args.o)