import h5py
import json
import os

MODEL_PATH = r"c:\Users\acer\Downloads\Telegram Desktop\ICD (2)\ICD\ml\bilstm_icd_model (1).h5"

if os.path.exists(MODEL_PATH):
    try:
        with h5py.File(MODEL_PATH, 'r') as f:
            print(f"H5 Keys: {list(f.keys())}")
            if 'model_config' in f.attrs:
                config = f.attrs['model_config']
                if isinstance(config, bytes):
                    config = config.decode('utf-8')
                config_json = json.loads(config)
                print("\nModel Config (Layers):")
                for layer in config_json['config']['layers']:
                    print(f"- {layer['class_name']}: {layer['config'].get('name', 'N/A')}")
                    if 'batch_input_shape' in layer['config']:
                        print(f"  Batch Input Shape: {layer['config']['batch_input_shape']}")
            
            if 'layer_names' in f.attrs:
                print(f"\nLayer Names: {f.attrs['layer_names']}")
    except Exception as e:
        print(f"Error reading H5: {e}")
else:
    print("Model file not found.")

PT_MODEL_PATH = r"c:\Users\acer\Downloads\Telegram Desktop\ICD (2)\ICD\ml\icd_model.pt"
if os.path.exists(PT_MODEL_PATH):
    print(f"\nPyTorch model found at {PT_MODEL_PATH}")
    import torch
    try:
        # Load state dict or full model
        try:
            checkpoint = torch.load(PT_MODEL_PATH, map_location='cpu')
            if isinstance(checkpoint, dict):
                print(f"PT Keys: {list(checkpoint.keys())}")
            else:
                print(f"PT Model Type: {type(checkpoint)}")
        except:
            print("PT load failed.")
    except Exception as e:
        print(f"Error checking PT model: {e}")
