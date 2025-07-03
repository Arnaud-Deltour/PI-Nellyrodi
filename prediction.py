import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import cv2

# Load CSV
#df = pd.read_csv("generated_color_palette_dataset.csv")
df = pd.read_csv("data_abstract2.csv")

X1 = df.drop(columns=['col2_h', 'col2_s', 'col2_v',
     'col3_h', 'col3_s', 'col3_v',
     'col4_h', 'col4_s', 'col4_v'])
X1.columns =["1","2","3"]
X = X1.values.astype(np.float32)
y1 = df.drop(columns=['col1_h', 'col1_s', 'col1_v'])
y1.columns = ["1","2","3","4","5","6","7","8","9"]
y = y1.values.astype(np.float32)
"""
X2 = df.drop(columns=['col3_h', 'col3_s', 'col3_v',
     'col4_h', 'col4_s', 'col4_v',
     'col1_h', 'col1_s', 'col1_v'])
X2.columns =["1","2","3"]
X = X2.values.astype(np.float32)
y2 = df.drop(columns=['col2_h', 'col2_s', 'col2_v'])
y2.columns = ["1","2","3","4","5","6","7","8","9"]
y = y2.values.astype(np.float32)

X3 = df.drop(columns=['col2_h', 'col2_s', 'col2_v',
     'col4_h', 'col4_s', 'col4_v',
     'col1_h', 'col1_s', 'col1_v'])
X3.columns =["1","2","3"]
X = X3.values.astype(np.float32)
y3 = df.drop(columns=['col3_h', 'col3_s', 'col3_v'])
y3.columns = ["1","2","3","4","5","6","7","8","9"]
y = y3.values.astype(np.float32)

X4 = df.drop(columns=['col3_h', 'col3_s', 'col3_v',
     'col2_h', 'col2_s', 'col2_v',
     'col1_h', 'col1_s', 'col1_v'])
X4.columns =["1","2","3"]
y4 = df.drop(columns=['col4_h', 'col4_s', 'col4_v'])
y4.columns = ["1","2","3","4","5","6","7","8","9"]

X = pd.concat([X1,X2,X3,X4]).values.astype(np.float32)
y = pd.concat([y1,y2,y3,y4]).values.astype(np.float32)
"""

#X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.01, random_state=103)
X_train, y_train = X, y
X_test, y_test = X[25:32], y[25:32]

model = Sequential([
    Dense(64, input_dim=3, activation='relu'),
    Dense(128, activation='relu'),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    #Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(9, activation='linear')  # 3 HSV vectors = 9 values
])

model.compile(optimizer='Adam', loss='mse', metrics=['mae','accuracy'])

model.fit(X_train, y_train, epochs=2000, verbose=1)
predictions = model.predict(X_test)


def show_input_and_palettes(inputs, predictions):
    num_samples = len(predictions)
    fig, axes = plt.subplots(num_samples, 1, figsize=(6, 1.5 * num_samples))

    if num_samples == 1:
        axes = [axes]  # Ensure iterable

    for i, (input_lab, pred) in enumerate(zip(inputs, predictions)):
        input_temp = np.clip(np.array(input_lab)*255,0,255).astype(int).reshape(1, 1, 3)
        pred_temp = np.clip(np.array(pred)*255,0,255).astype(int).reshape(1, 3, 3)

        # Concatenate input + prediction colors: shape (1, 4, 3)
        row_colors = np.concatenate([input_temp, pred_temp], axis=1)

        axes[i].imshow(cv2.cvtColor(np.array(row_colors, dtype=np.uint8), cv2.COLOR_LAB2RGB))
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_title(f"Input + Predicted Palette {i+1}")

    plt.tight_layout()
    plt.show()

show_input_and_palettes(X_test, predictions)

#print(model.evaluate(X_test, y_test))

model.save('abstract_art.keras')

"""
palettes = np.concatenate([X_train[1],y_train[1]], axis=0)
pal = palettes*255
pal1 = np.clip(pal,0,255).astype(int).reshape(4, 3).reshape(1, 4, 3)
pal_lab = cv2.cvtColor(np.array(pal1, dtype=np.uint8), cv2.COLOR_LAB2RGB)

plt.imshow(pal_lab)
plt.show()"""