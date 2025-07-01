import numpy as np
import pandas as pd

def generate_dataset(num_samples=1000):
    X = []
    y = []

    for _ in range(num_samples):
        # Random input RGB color (normalized)
        base_color = np.random.rand(3)

        # Generate a palette based on the base color:
        # Color 1: slightly brighter
        # Color 2: slightly darker
        # Color 3: shifted hue or blended
        color1 = np.clip(base_color + np.random.uniform(0.05, 0.2, 3), 0, 1)
        color2 = np.clip(base_color - np.random.uniform(0.05, 0.2, 3), 0, 1)
        color3 = np.clip((base_color + np.roll(base_color, 1)) / 2, 0, 1)

        # Store input and flattened target (9 values: 3 RGB vectors)
        X.append(base_color)
        y.append(np.concatenate([color1, color2, color3]))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X, y = generate_dataset(2000)

# Combine input and output for saving
data = np.hstack([X, y])

# Define column names
columns = [
    'col1_h', 'col1_s', 'col1_v',
    'col2_h', 'col2_s', 'col2_v',
    'col3_h', 'col3_s', 'col3_v',
    'col4_h', 'col4_s', 'col4_v'
]

# Create and save DataFrame
df = pd.DataFrame(data, columns=columns)
df.to_csv("generated_color_palette_dataset.csv", index=False)