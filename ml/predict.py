import os
import sys
import pandas as pd

# Add the ml_service directory to path
service_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ml_service_path = os.path.join(service_path, "ml_service")
sys.path.append(ml_service_path)

from src.detector import ICD10Detector

# Initialize the detector once (using full dataset)
data_path = os.path.join(ml_service_path, "data", "full_raw_codes.csv")
detector = ICD10Detector(data_path)

def predict_icd(description):
    """
    Main entry point for Django views.
    Integrates the BiLSTM-Hybrid detector logic.
    """
    try:
        # Get top 3 results from our new detector
        raw_results = detector.detect(description, top_k=1)
        
        # Map to the format expected by the Django template/JSON view
        formatted_results = []
        for res in raw_results:
            formatted_results.append({
                "icd_code": res["code"],
                "description": res["description"],
                "confidence": res["score"],
                "match_type": "hybrid_bilstm",
                "reasoning": res["reasoning"]
            })
            
        return formatted_results
    except Exception as e:
        print(f"Error in hybrid prediction: {e}")
        return []

if __name__ == "__main__":
    # Test
    test_desc = "The patient presents with gastrointestinal illness caused by foodborne Clostridium perfringens intoxication."
    print(predict_icd(test_desc))
