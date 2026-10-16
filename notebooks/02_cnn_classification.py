from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix
 

# ============================================================
# PHASE 2
# CNN-BASED POTATO LEAF DISEASE CLASSIFICATION
# ============================================================

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_ROOT / "data" / "train"
VAL_DIR = PROJECT_ROOT / "data" / "val"
TEST_DIR = PROJECT_ROOT / "data" / "test"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("PHASE 2 - CNN CLASSIFICATION")
print("=" * 70)

print(f"Project root : {PROJECT_ROOT}")
print(f"Train       : {TRAIN_DIR}")
print(f"Validation  : {VAL_DIR}")
print(f"Test        : {TEST_DIR}")


# ------------------------------------------------------------
# 2. BASIC SETTINGS
# ------------------------------------------------------------

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

print("\nSettings:")
print(f"Image size : {IMAGE_SIZE}")
print(f"Batch size : {BATCH_SIZE}")
print(f"Epochs     : {EPOCHS}")


# ------------------------------------------------------------
# 3. LOAD DATASETS
# ------------------------------------------------------------

print("\nLoading training dataset...")

train_ds = image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

print("\nLoading validation dataset...")

val_ds = image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nLoading test dataset...")

test_ds = image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ------------------------------------------------------------
# 4. CLASS NAMES
# ------------------------------------------------------------

class_names = train_ds.class_names

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

NUM_CLASSES = len(class_names)

print(f"\nNumber of classes: {NUM_CLASSES}")


# Save class names
with open(RESULTS_DIR / "class_names.json", "w") as f:
    json.dump(class_names, f, indent=4)


# ------------------------------------------------------------
# 5. DATA AUGMENTATION
# ------------------------------------------------------------

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.10),
], name="data_augmentation")


# ------------------------------------------------------------
# 6. PREFETCH DATA
# ------------------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)


# ------------------------------------------------------------
# 7. BUILD CNN MODEL
# ------------------------------------------------------------

print("\nBuilding CNN model...")

model = models.Sequential([
    
    # Input
    layers.Input(shape=(224, 224, 3)),

    # Data augmentation
    data_augmentation,

    # Normalize pixels from [0,255] to [0,1]
    layers.Rescaling(1.0 / 255),

    # -------------------------
    # Convolution Block 1
    # -------------------------
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    # -------------------------
    # Convolution Block 2
    # -------------------------
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    # -------------------------
    # Convolution Block 3
    # -------------------------
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    # -------------------------
    # Convolution Block 4
    # -------------------------
    layers.Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    # -------------------------
    # Classification Head
    # -------------------------
    layers.Flatten(),

    layers.Dense(128, activation="relu"),

    layers.Dropout(0.5),

    layers.Dense(NUM_CLASSES, activation="softmax")
])


# ------------------------------------------------------------
# 8. DISPLAY MODEL
# ------------------------------------------------------------

model.summary()


# ------------------------------------------------------------
# 9. COMPILE MODEL
# ------------------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ------------------------------------------------------------
# 10. CALLBACKS
# ------------------------------------------------------------

callbacks = [

    # Save best model
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / "best_cnn_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    # Stop if validation performance stops improving
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    # Reduce learning rate if validation loss stops improving
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]


# ------------------------------------------------------------
# 11. TRAIN MODEL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STARTING CNN TRAINING")
print("=" * 70)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ------------------------------------------------------------
# 12. SAVE FINAL MODEL
# ------------------------------------------------------------

final_model_path = MODEL_DIR / "potato_disease_cnn_final.keras"

model.save(final_model_path)

print("\nFinal model saved at:")
print(final_model_path)


# ------------------------------------------------------------
# 13. PLOT TRAINING ACCURACY
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title("CNN Training and Validation Accuracy")

plt.legend()
plt.grid(True)

plt.savefig(
    RESULTS_DIR / "training_validation_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 14. PLOT TRAINING LOSS
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("CNN Training and Validation Loss")

plt.legend()
plt.grid(True)

plt.savefig(
    RESULTS_DIR / "training_validation_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 15. TEST EVALUATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EVALUATING ON TEST DATA")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(test_ds)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# ------------------------------------------------------------
# 16. GET PREDICTIONS
# ------------------------------------------------------------

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ------------------------------------------------------------
# 17. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print(report)

with open(
    RESULTS_DIR / "classification_report.txt",
    "w"
) as f:
    f.write(report)


# ------------------------------------------------------------
# 18. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ------------------------------------------------------------
# 19. PLOT CONFUSION MATRIX
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.imshow(cm)

plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")

plt.xticks(
    range(NUM_CLASSES),
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(NUM_CLASSES),
    class_names
)

# Write values inside cells
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 20. SAMPLE PREDICTIONS
# ------------------------------------------------------------

print("\nGenerating sample prediction image...")

plt.figure(figsize=(12, 9))

for images, labels in test_ds.take(1):

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    for i in range(min(9, len(images))):

        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(
            images[i].numpy().astype("uint8")
        )

        true_label = class_names[
            labels[i].numpy()
        ]

        predicted_label = class_names[
            predicted_classes[i]
        ]

        plt.title(
            f"True: {true_label}\nPred: {predicted_label}"
        )

        plt.axis("off")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "sample_predictions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 21. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PHASE 2 COMPLETED")
print("=" * 70)

print(f"""
Classes:
{class_names}

Test Accuracy:
{test_accuracy * 100:.2f}%

Model:
{final_model_path}

Results saved in:
{RESULTS_DIR}

Generated files:
- training_validation_accuracy.png
- training_validation_loss.png
- confusion_matrix.png
- sample_predictions.png
- classification_report.txt
- class_names.json
""")

print("=" * 70)
print("NEXT PHASE: YOLO-BASED DISEASE LOCALIZATION")
print("=" * 70)