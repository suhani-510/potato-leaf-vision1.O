from pathlib import Path
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import hashlib
import shutil
import random

 
# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_DIR / "data" / "raw"

TRAIN_DIR = PROJECT_DIR / "data" / "train"
VAL_DIR = PROJECT_DIR / "data" / "val"
TEST_DIR = PROJECT_DIR / "data" / "test"

RESULTS_DIR = PROJECT_DIR / "results"


# ============================================================
# 2. CREATE FOLDERS
# ============================================================

for folder in [
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    RESULTS_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. CLASSES
# ============================================================

CLASSES = [
    "healthy",
    "early_blight",
    "late_blight"
]

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# 4. START
# ============================================================

print("\n" + "=" * 60)
print("POTATO LEAF VISION")
print("PHASE 1 - DATASET ANALYSIS")
print("=" * 60)


# ============================================================
# 5. CHECK RAW DATASET
# ============================================================

if not RAW_DIR.exists():

    print("\n❌ ERROR: data/raw folder not found.")

    print("\nExpected structure:")
    print("""
potato-leaf-vision/
│
├── data/
│   └── raw/
│       ├── healthy/
│       ├── early_blight/
│       └── late_blight/
│
└── notebooks/
    └── 01_dataset_analysis.py
""")

    input("\nPress Enter to exit...")
    exit()


# ============================================================
# 6. CHECK CLASS FOLDERS
# ============================================================

print("\nChecking class folders...\n")

missing = False

for class_name in CLASSES:

    folder = RAW_DIR / class_name

    if folder.exists():

        print(f"✅ {class_name}")

    else:

        print(f"❌ {class_name} folder missing")
        missing = True


if missing:

    print("\nPlease create the missing folders inside:")

    print(RAW_DIR)

    input("\nPress Enter to exit...")
    exit()


# ============================================================
# 7. SCAN IMAGES
# ============================================================

print("\nScanning images...")

records = []
bad_images = []


for class_name in CLASSES:

    class_folder = RAW_DIR / class_name

    for image_path in class_folder.rglob("*"):

        if image_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        try:

            with Image.open(image_path) as img:

                width, height = img.size

                # Actually load image
                img.load()

            records.append({
                "filename": image_path.name,
                "class": class_name,
                "width": width,
                "height": height,
                "path": str(image_path)
            })

        except Exception as error:

            bad_images.append(str(image_path))

            print(
                f"⚠️ Bad image: {image_path.name}"
            )


# ============================================================
# 8. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(records)


if df.empty:

    print("\n❌ No valid images found.")

    print("\nCheck this folder:")

    print(RAW_DIR)

    input("\nPress Enter to exit...")
    exit()


print(
    f"\n✅ Total valid images: {len(df)}"
)


# ============================================================
# 9. CLASS DISTRIBUTION
# ============================================================

print("\n" + "-" * 60)
print("CLASS DISTRIBUTION")
print("-" * 60)

class_counts = df["class"].value_counts()

print(class_counts)


# ============================================================
# 10. IMAGE DIMENSIONS
# ============================================================

print("\n" + "-" * 60)
print("IMAGE DIMENSIONS")
print("-" * 60)

print(
    df[
        ["width", "height"]
    ].describe()
)


# ============================================================
# 11. CORRUPTED IMAGES
# ============================================================

print("\n" + "-" * 60)
print("CORRUPTED IMAGES")
print("-" * 60)

if len(bad_images) == 0:

    print("✅ No corrupted images found.")

else:

    print(
        f"⚠️ {len(bad_images)} corrupted images found."
    )

    for image in bad_images:
        print(image)


# ============================================================
# 12. DUPLICATE CHECK
# ============================================================

print("\n" + "-" * 60)
print("CHECKING DUPLICATES")
print("-" * 60)


def calculate_hash(path):

    try:

        with open(path, "rb") as file:

            return hashlib.md5(
                file.read()
            ).hexdigest()

    except Exception:

        return None


df["hash"] = df["path"].apply(
    calculate_hash
)


duplicate_rows = df[
    df.duplicated(
        subset=["hash"],
        keep=False
    )
]


if duplicate_rows.empty:

    print("✅ No exact duplicates found.")

else:

    print(
        f"⚠️ Duplicate images: "
        f"{len(duplicate_rows)}"
    )

    print(
        duplicate_rows[
            [
                "filename",
                "class",
                "path"
            ]
        ].to_string(index=False)
    )


# ============================================================
# 13. SAVE DATASET INFORMATION
# ============================================================

dataset_info = (
    RESULTS_DIR /
    "dataset_information.csv"
)

df.to_csv(
    dataset_info,
    index=False
)

print(
    f"\n✅ Dataset information saved:"
)

print(dataset_info)


# ============================================================
# 14. CLASS DISTRIBUTION GRAPH
# ============================================================

print("\nCreating class distribution graph...")

plt.figure(
    figsize=(8, 5)
)

class_counts.plot(
    kind="bar"
)

plt.title(
    "Potato Leaf Dataset - Class Distribution"
)

plt.xlabel("Class")

plt.ylabel("Number of Images")

plt.xticks(
    rotation=0
)

plt.tight_layout()

graph_path = (
    RESULTS_DIR /
    "class_distribution.png"
)

plt.savefig(
    graph_path,
    dpi=300
)

plt.close()

print(
    f"✅ Saved: {graph_path}"
)


# ============================================================
# 15. IMAGE DIMENSION GRAPH
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.scatter(
    df["width"],
    df["height"],
    alpha=0.5
)

plt.title(
    "Image Dimensions"
)

plt.xlabel("Width")

plt.ylabel("Height")

plt.tight_layout()

dimension_path = (
    RESULTS_DIR /
    "image_dimensions.png"
)

plt.savefig(
    dimension_path,
    dpi=300
)

plt.close()

print(
    f"✅ Saved: {dimension_path}"
)


# ============================================================
# 16. SAMPLE IMAGES
# ============================================================

print("\nCreating sample image visualization...")


fig, axes = plt.subplots(
    1,
    len(CLASSES),
    figsize=(15, 5)
)


for index, class_name in enumerate(CLASSES):

    class_data = df[
        df["class"] == class_name
    ]

    if len(class_data) == 0:

        axes[index].set_title(
            class_name
        )

        axes[index].axis("off")

        continue


    image_path = Path(
        class_data.iloc[0]["path"]
    )

    image = Image.open(
        image_path
    )

    axes[index].imshow(
        image
    )

    axes[index].set_title(
        class_name.replace(
            "_",
            " "
        ).title()
    )

    axes[index].axis("off")


plt.tight_layout()

sample_path = (
    RESULTS_DIR /
    "sample_images.png"
)

plt.savefig(
    sample_path,
    dpi=300
)

plt.close()

print(
    f"✅ Saved: {sample_path}"
)


# ============================================================
# 17. REMOVE CORRUPTED IMAGES FROM DATAFRAME
# ============================================================

clean_df = df.copy()

if bad_images:

    clean_df = clean_df[
        ~clean_df["path"].isin(
            bad_images
        )
    ]


# ============================================================
# 18. MANUAL STRATIFIED SPLIT
# ============================================================
#
# We are intentionally NOT using sklearn here.
#
# 70% train
# 15% validation
# 15% test
#
# ============================================================

print("\n" + "-" * 60)
print("CREATING DATASET SPLIT")
print("-" * 60)


random.seed(42)


train_rows = []
val_rows = []
test_rows = []


for class_name in CLASSES:

    class_data = clean_df[
        clean_df["class"] == class_name
    ].copy()

    rows = class_data.to_dict(
        "records"
    )

    random.shuffle(rows)

    total = len(rows)

    train_end = int(
        total * 0.70
    )

    val_end = int(
        total * 0.85
    )

    train_rows.extend(
        rows[:train_end]
    )

    val_rows.extend(
        rows[
            train_end:val_end
        ]
    )

    test_rows.extend(
        rows[
            val_end:
        ]
    )


train_df = pd.DataFrame(
    train_rows
)

val_df = pd.DataFrame(
    val_rows
)

test_df = pd.DataFrame(
    test_rows
)


print(
    f"Training:   {len(train_df)}"
)

print(
    f"Validation: {len(val_df)}"
)

print(
    f"Testing:    {len(test_df)}"
)


# ============================================================
# 19. COPY IMAGES
# ============================================================

def copy_images(
    dataframe,
    destination
):

    for class_name in CLASSES:

        class_destination = (
            destination /
            class_name
        )

        class_destination.mkdir(
            parents=True,
            exist_ok=True
        )

        class_data = dataframe[
            dataframe["class"] ==
            class_name
        ]

        for _, row in class_data.iterrows():

            source = Path(
                row["path"]
            )

            target = (
                class_destination /
                source.name
            )

            shutil.copy2(
                source,
                target
            )


print("\nCopying training images...")

copy_images(
    train_df,
    TRAIN_DIR
)


print("Copying validation images...")

copy_images(
    val_df,
    VAL_DIR
)


print("Copying testing images...")

copy_images(
    test_df,
    TEST_DIR
)


# ============================================================
# 20. SAVE SPLIT INFORMATION
# ============================================================

train_df.to_csv(
    RESULTS_DIR /
    "train.csv",
    index=False
)

val_df.to_csv(
    RESULTS_DIR /
    "validation.csv",
    index=False
)

test_df.to_csv(
    RESULTS_DIR /
    "test.csv",
    index=False
)


# ============================================================
# 21. PRINT FINAL DISTRIBUTION
# ============================================================

print("\n" + "-" * 60)
print("FINAL SPLIT")
print("-" * 60)


print("\nTRAIN:")

print(
    train_df["class"]
    .value_counts()
)


print("\nVALIDATION:")

print(
    val_df["class"]
    .value_counts()
)


print("\nTEST:")

print(
    test_df["class"]
    .value_counts()
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("✅ PHASE 1 COMPLETED")
print("=" * 60)

print(
    f"""
Total images: {len(clean_df)}

Train:       {len(train_df)}
Validation:  {len(val_df)}
Test:        {len(test_df)}

Classes:
    • Healthy
    • Early Blight
    • Late Blight

Results:
    {RESULTS_DIR}

Train:
    {TRAIN_DIR}

Validation:
    {VAL_DIR}

Test:
    {TEST_DIR}
"""
)

print("=" * 60)
print(
    "NEXT: PHASE 2 - CNN CLASSIFICATION"
)
print("=" * 60)