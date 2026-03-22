import pandas as pd
import os

def main():
    input_file = r"f:\Dachuang\Nee_Prediction\tcn_informer_backup\tcn+informer\inf\data\raw_data\smear2008-18_allmonths.csv"
    output_dir = r"f:\Dachuang\Nee_Prediction\tcn_informer_backup\tcn+informer\inf\data\change_data"
    output_file = os.path.join(output_dir, "smear_original_freq_processed.csv")

    print(f"Reading from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Original shape: {df.shape}")

    # Drop Unnamed column if it exists
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    # Convert Time to datetime
    df['Time'] = pd.to_datetime(df['Time'])

    # Rename Time to date and NEE to target for Informer
    df.rename(columns={
        'Time': 'date',
        'NEE': 'target'
    }, inplace=True)

    # Sort by date just to be safe
    df.sort_values(by='date', inplace=True)

    # Format date to standard string format expected by Informer (YYYY-MM-DD HH:MM:SS)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Interpolate lightly for any internal missing values to keep the time-series unbroken
    # Limit direction to ensure we don't carry values across huge gaps
    df.interpolate(method='linear', limit=5, inplace=True)
    
    # Drop rows that still have NaNs (e.g. leading or very large gaps)
    df.dropna(inplace=True)

    # Round to sensible precision to save space
    df = df.round(5)

    # Reorder columns: date first, target last
    cols = list(df.columns)
    cols.remove('date')
    cols.remove('target')
    final_cols = ['date'] + cols + ['target']
    df = df[final_cols]

    print(f"Final shape (Original Frequency): {df.shape}")
    
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Success! Original frequency output saved to: {output_file}")
    
if __name__ == "__main__":
    main()
