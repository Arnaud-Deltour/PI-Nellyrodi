import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# Load CSV
df = pd.read_csv("generated_color_palette_dataset.csv")
X = df[['input_r', 'input_g', 'input_b']].values.astype(np.float32)
y = df[
    ['out1_r', 'out1_g', 'out1_b',
     'out2_r', 'out2_g', 'out2_b',
     'out3_r', 'out3_g', 'out3_b']
].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=1)



model = Sequential([
    Dense(32, input_dim=3, activation='relu'),
    Dense(64, activation='relu'),
    Dense(9, activation='sigmoid')  # 3 RGB vectors = 9 values
])

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=100, verbose=1)
predictions = model.predict(X_test)

predicted_rgb = (predictions * 255).astype(int)


def show_input_and_palettes(inputs, predictions):
    num_samples = len(predictions)
    fig, axes = plt.subplots(num_samples, 1, figsize=(6, 1.5 * num_samples))

    if num_samples == 1:
        axes = [axes]  # Ensure iterable

    for i, (input_rgb, pred) in enumerate(zip(inputs, predictions)):
        input_color = (input_rgb * 255).astype(np.uint8).reshape(1, 1, 3)
        pred_palette = (pred.reshape(3, 3) * 255).astype(np.uint8).reshape(1, 3, 3)

        # Concatenate input + prediction colors: shape (1, 4, 3)
        row_colors = np.concatenate([input_color, pred_palette], axis=1)

        axes[i].imshow(row_colors)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_title(f"Input + Predicted Palette {i+1}")

    plt.tight_layout()
    plt.show()

show_input_and_palettes(X_test, predictions)
