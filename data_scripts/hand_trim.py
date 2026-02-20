import pandas as pd
import os

def trim_single_csv(input_path, n_rows, output_folder):
    """
    Reads a single CSV, removes the first n_rows of data (keeping the header),
    and saves it to the output_folder.
    """
    # 1. Validation
    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}")
        return

    filename = os.path.basename(input_path)
    
    # 2. Create output folder if needed
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    output_path = os.path.join(output_folder, filename)

    print(f"--- Processing File: {filename} ---")
    print(f"  Input:  {input_path}")
    print(f"  Removing first {n_rows} rows...")

    try:
        # 3. Read and Trim
        # skiprows=range(1, n_rows + 1) keeps the header (row 0) but skips rows 1 to n
        df = pd.read_csv(input_path, skiprows=range(1, n_rows + 1))
        
        # 4. Save
        df.to_csv(output_path, index=False)
        print(f"  [SUCCESS] Saved to: {output_path}")
        print(f"  Remaining data rows: {len(df)}")
        
    except Exception as e:
        print(f"  [ERROR] Failed to process file: {e}")

if __name__ == "__main__":
    # ==========================================
    # EDIT YOUR SETTINGS HERE
    # ==========================================
    
    # Path to the specific file you want to trim
    FILE_TO_TRIM = "/home/tt/ind_ml_rawan_teresa/AA-MAIN-FOLDER/Yan_all_data/Yan_combos_sensor/Yan_combo4_2_CapacitanceTable.csv"
    
    # How many rows to delete from the start
    ROWS_TO_REMOVE = 570
    
    # Where to save the new file
    OUTPUT_DIR = "/home/tt/ind_ml_rawan_teresa/AA-MAIN-FOLDER/Yan_all_data/Yan_combos_sensor_trimed"
    
    # ==========================================
    
    trim_single_csv(FILE_TO_TRIM, ROWS_TO_REMOVE, OUTPUT_DIR)