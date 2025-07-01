import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import cv2

# Load CSV
#df = pd.read_csv("generated_color_palette_dataset.csv")
df = pd.read_csv("data2.csv")
print(df)
X1 = df.drop(columns=['col2_h', 'col2_s', 'col2_v',
     'col3_h', 'col3_s', 'col3_v',
     'col4_h', 'col4_s', 'col4_v'])
X1.columns =["1","2","3"]
y1 = df.drop(columns=['col1_h', 'col1_s', 'col1_v'])
y1.columns = ["1","2","3","4","5","6","7","8","9"]
X2 = df.drop(columns=['col3_h', 'col3_s', 'col3_v',
     'col4_h', 'col4_s', 'col4_v',
     'col1_h', 'col1_s', 'col1_v'])
X2.columns =["1","2","3"]
y2 = df.drop(columns=['col2_h', 'col2_s', 'col2_v'])
y2.columns = ["1","2","3","4","5","6","7","8","9"]
X3 = df.drop(columns=['col2_h', 'col2_s', 'col2_v',
     'col4_h', 'col4_s', 'col4_v',
     'col1_h', 'col1_s', 'col1_v'])
X3.columns =["1","2","3"]
y3 = df.drop(columns=['col3_h', 'col3_s', 'col3_v'])
y3.columns = ["1","2","3","4","5","6","7","8","9"]
X4 = df.drop(columns=['col3_h', 'col3_s', 'col3_v',
     'col2_h', 'col2_s', 'col2_v',
     'col1_h', 'col1_s', 'col1_v'])
X4.columns =["1","2","3"]
y4 = df.drop(columns=['col4_h', 'col4_s', 'col4_v'])
y4.columns = ["1","2","3","4","5","6","7","8","9"]

X = pd.concat([X1,X2,X3,X4]).values.astype(np.float32)
y = pd.concat([y1,y2,y3,y4]).values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=10)


model = Sequential([
    Dense(32, input_dim=3, activation='relu'),
    Dense(64, activation='relu'),
    Dense(9, activation='sigmoid')  # 3 HSV vectors = 9 values
])

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, verbose=1)
predictions = model.predict(X_test)


def show_input_and_palettes(inputs, predictions):
    num_samples = len(predictions)
    fig, axes = plt.subplots(num_samples, 1, figsize=(6, 1.5 * num_samples))

    if num_samples == 1:
        axes = [axes]  # Ensure iterable

    for i, (input_hsv, pred) in enumerate(zip(inputs, predictions)):
        input_rgb = np.array([input_hsv[0]*179, input_hsv[1]*255, input_hsv[2]*255])
        input_color = input_rgb.astype(np.uint8).reshape(1, 1, 3)
        pred_rgb = np.array([pred[0]*179, pred[1]*255, pred[2]*255, pred[3]*179, pred[4]*255, pred[5]*255, pred[6]*179, pred[7]*255, pred[8]*255])
        pred_palette = pred_rgb.reshape(3, 3).astype(np.uint8).reshape(1, 3, 3)

        # Concatenate input + prediction colors: shape (1, 4, 3)
        row_colors = np.concatenate([input_color, pred_palette], axis=1)

        axes[i].imshow(cv2.cvtColor(row_colors, cv2.COLOR_HSV2RGB))
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_title(f"Input + Predicted Palette {i+1}")

    plt.tight_layout()
    plt.show()

show_input_and_palettes(X_test, predictions)

print(model.evaluate(X_test, y_test))

model.save('impressionist_paintings.keras')
