import pandas as pd

def load_data():
    # Simulate loading data from a CSV or database
    data = {
        "c_id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "score": [95, 88, 76]
    }
    df = pd.DataFrame(data)
    return df
