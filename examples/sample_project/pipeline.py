from data_loader import load_data

def run_pipeline():
    df = load_data()
    print("Data Loaded:")
    print(df)

    # Add processing logic
    df["passed"] = df["score"] > 80

    print("\nProcessed Data:")
    print(df)

if __name__ == "__main__":
    run_pipeline()
