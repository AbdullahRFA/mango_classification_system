# 📦 Import necessary libraries
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import seaborn as sns

# 📁 Define paths to dataset, model, and label file
DATA_DIR = "data/mango_dataset_resized"
MODEL_PATH = "mango_model.h5"
LABELS_PATH = "labels.txt"
IMG_SIZE = (224, 224)  # Target image size for model input

# 🔍 Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# 📄 Load class labels from text file
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# 🖼️ Prepare one sample image per class for visual overview
class_images = []         # Store loaded images
class_predictions = []    # Store predicted class names
class_confidences = []    # Store prediction confidence scores

for label in labels:
    class_dir = os.path.join(DATA_DIR, label)         # Path to class folder
    image_files = os.listdir(class_dir)               # List of image files in class folder
    if not image_files:
        continue                                      # Skip if folder is empty
    img_path = os.path.join(class_dir, image_files[0])  # Use first image in folder
    img = load_img(img_path, target_size=IMG_SIZE)      # Load and resize image
    img_array = img_to_array(img) / 255.0               # Normalize pixel values
    class_images.append(img)                            # Store image for display

    pred = model.predict(np.expand_dims(img_array, axis=0))[0]  # Predict class probabilities
    pred_idx = np.argmax(pred)                                  # Get predicted class index
    confidence = pred[pred_idx]                                 # Get confidence score
    class_predictions.append(labels[pred_idx])                  # Store predicted label
    class_confidences.append(confidence)                        # Store confidence

# 🧪 Prepare test data using validation split (20% of data)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

test_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=32,
    subset="validation",         # Use validation split as test set
    class_mode="categorical",    # One-hot encoded labels
    shuffle=False                # Keep order for evaluation
)

# 🔮 Predict on test data
pred_probs = model.predict(test_gen)                     # Get predicted probabilities
pred_classes = np.argmax(pred_probs, axis=1)             # Convert to predicted class indices
true_classes = test_gen.classes                          # Get true class indices

# 📊 Generate classification report dynamically
report = classification_report(true_classes, pred_classes, target_names=labels, output_dict=True)
report_df = pd.DataFrame(report).T                       # Convert report to DataFrame
if "support" in report_df.columns:                       # Drop 'support' column if present
    report_df = report_df.drop(columns=["support"])
report_df = report_df.round(2)                           # Round metrics for display
report_df.index = [label.title() if label in labels else label for label in report_df.index]  # Format index

# 🔍 Generate confusion matrix
conf_matrix = confusion_matrix(true_classes, pred_classes)

# ✅ Calculate overall accuracy
accuracy_value = accuracy_score(true_classes, pred_classes)

# 🍋 Window 1: Show one image per class with prediction confidence
fig1 = plt.figure(figsize=(16, 10))
fig1.suptitle("🍋 Mango Classification Overview", fontsize=20)

for i in range(len(class_images)):
    ax = fig1.add_subplot(2, 4, i + 1)                   # Create subplot for each image
    ax.imshow(class_images[i])                          # Display image
    ax.axis('off')                                      # Hide axis
    ax.set_title(f"{class_predictions[i]}\nConfidence: {class_confidences[i] * 100:.2f}%", fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])                   # Adjust layout
plt.show()                                               # Show image overview window

# 📊 Window 2: Show confusion matrix and classification report
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle("📊 Mango Model Evaluation", fontsize=18)

# 🔍 Plot confusion matrix as heatmap
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax1)
ax1.set_title("🔍 Confusion Matrix", fontsize=14)
ax1.set_xlabel("Predicted Label")
ax1.set_ylabel("True Label")

# 📋 Display classification report as table
table = ax2.table(cellText=report_df.values,
                  colLabels=report_df.columns,
                  rowLabels=report_df.index,
                  loc='center',
                  cellLoc='center')
table.scale(1.0, 1.5)                                    # Resize table
table.auto_set_font_size(False)
table.set_fontsize(9)
ax2.axis('off')                                          # Hide axis
ax2.set_title("📋 Classification Report", fontsize=14)

# ✅ Display total accuracy below the table
ax2.text(0.5, -0.15, f"✅ Total Accuracy: {accuracy_value * 100:.2f}%",
         fontsize=12, ha='center', va='center', transform=ax2.transAxes, color='green')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])                # Adjust layout
plt.show()                                               # Show evaluation window





# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.metrics import classification_report, confusion_matrix
#
# # 📁 Paths
# DATA_DIR = "data/mango_dataset_resized"
# MODEL_PATH = "mango_model.h5"
# LABELS_PATH = "labels.txt"
#
# # 🖼️ Image settings
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 32
#
# # 🔍 Load trained model
# model = tf.keras.models.load_model(MODEL_PATH)
#
# # 📄 Load class labels
# with open(LABELS_PATH, "r") as f:
#     labels = [line.strip() for line in f.readlines()]
#
# # 🧪 Prepare test data using validation split
# datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
#
# test_generator = datagen.flow_from_directory(
#     DATA_DIR,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode="categorical",
#     subset="validation",
#     shuffle=False
# )
#
# # 🔮 Make predictions
# pred_probs = model.predict(test_generator)
# pred_classes = np.argmax(pred_probs, axis=1)
# true_classes = test_generator.classes
#
# # 📊 Print classification report
# print("\n📊 Classification Report:")
# print(classification_report(true_classes, pred_classes, target_names=labels))
#
# # 🔍 Print confusion matrix
# print("\n🔍 Confusion Matrix:")
# print(confusion_matrix(true_classes, pred_classes))
