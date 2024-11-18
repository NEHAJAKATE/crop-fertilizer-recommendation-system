import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensorflow_probability as tfp
import tensorflow as tf

# Load data
data = pd.read_csv('/content/Crop_recommendation.csv')

# Display the first few rows
print(data.head())

# Check column names
print(data.columns)

# Encode labels
data['encoded_label'] = data['label'].astype('category').cat.codes
print(data.head())

# Split features and labels
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['encoded_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to NumPy arrays for TensorFlow
X_train = X_train.values.astype(np.float32)
X_test = X_test.values.astype(np.float32)
y_train = y_train.values.astype(np.float32).reshape(-1, 1)
y_test = y_test.values.astype(np.float32).reshape(-1, 1)

# Build the model
def build_bpr_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(128, activation='relu')(inputs)  
    x = Dense(64, activation='relu')(x)       
    outputs = Dense(1, activation='sigmoid')(x)  
    model = Model(inputs=inputs, outputs=outputs)
    return model

input_shape = (X_train.shape[1],)
model = build_bpr_model(input_shape)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Recommendation function
def recommend_fertilizer(crop_features):
    prob = model.predict(crop_features)
    return 'Recommended_Fertilizer' if prob > 0.5 else 'Alternative_Fertilizer'

# Test the recommendation
crop_features = np.array([[90, 42, 43, 20.8797, 82.0028, 6.5029, 202.9355]]).astype(np.float32)
recommendation = recommend_fertilizer(crop_features)
print(recommendation)

# Visualization: Pie chart
fertilizer_counts = data['label'].value_counts()
plt.figure(figsize=(10, 6))
plt.pie(fertilizer_counts, labels=fertilizer_counts.index, autopct='%1.1f%%')
plt.title('Distribution of Crops')
plt.show()

# Visualization: Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='ph', y='rainfall', hue='label')
plt.title('Scatter Plot of pH vs Rainfall')
plt.show()

# Pairplot
sns.pairplot(data, hue='label', vars=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Features')
plt.show()

# Box plots
plt.figure(figsize=(12, 10))
for i, col in enumerate(X.columns):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(x='label', y=col, data=data)
    plt.title(f'Box Plot of {col} by Crop')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Distribution histograms
data.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
plt.show()

# Violin plots
plt.figure(figsize=(12, 10))
for i, col in enumerate(X.columns):
    plt.subplot(3, 3, i + 1)
    sns.violinplot(x='label', y=col, data=data)
    plt.title(f'Violin Plot of {col} by Crop')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Training loss plot
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
