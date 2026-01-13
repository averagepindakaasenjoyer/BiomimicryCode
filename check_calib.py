import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

# Load stereo calibration for each distance
distances = {
    '8CM': os.getenv('STEREO_PARAMS_8CM'),
    '12CM': os.getenv('STEREO_PARAMS_12CM'),
    '16CM': os.getenv('STEREO_PARAMS_16CM')
}

print("Calibration Quality Check")
print("=" * 60)

for name, path in distances.items():
    if os.path.exists(path):
        data = np.load(path)
        print(f'\n{name} Calibration:')
        print(f"  Path: {path}")
        
        if 'stereo_error' in data:
            err = float(data['stereo_error'])
            print(f"  Stereo reprojection error: {err:.6f} pixels")
            if err < 0.5:
                print(f"    Status: EXCELLENT")
            elif err < 1.0:
                print(f"    Status: GOOD")
            elif err < 2.0:
                print(f"    Status: ACCEPTABLE")
            else:
                print(f"    Status: POOR (>2.0)")
        
        if 'R' in data:
            T = data['T']
            baseline = np.linalg.norm(T)
            print(f"  Baseline: {baseline:.6f} m ({baseline*100:.2f} cm)")
        
        print(f"  Keys: {list(data.keys())}")
    else:
        print(f'\n{name}: NOT FOUND at {path}')

print("\n" + "=" * 60)
print("Recommendation: Use calibration with lowest reprojection error")
