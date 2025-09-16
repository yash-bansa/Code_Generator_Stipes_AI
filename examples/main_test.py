from src.ingestion.read_csv import read_csv
from src.ingestion.validate_data import validate_data
from src.processing.transform import transform_data
from src.processing.feature_engineering import add_features
from src.ai_model.train_model import train_model
from src.ai_model.predict import make_predictions

def run_pipeline():
    # Read data
    df = read_csv("data/raw/sample_data.csv")
    
    # Validate
    df = validate_data(df)
    
    # Transform
    df = transform_data(df)
    
    # Feature engineering
    df = add_features(df, ["value"])
    
    # Train model
    model = train_model(df)
    
    # Predict
    predictions = make_predictions(model, df)
    predictions.show()

if __name__ == "__main__":
    run_pipeline()
