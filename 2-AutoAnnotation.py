import os
import random
import shutil
from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO

# --- Ontology Definition ---
ontology = CaptionOntology({
    "basketball players wearing navy blue jersey singlet USA": "USA Player",
    "basketball players wearing majority white jersey singlet": "Opponent Player",
    "Basketball orange ball": "Basketball",
    "Person with grey shirt and black pants and black shoes": "Referee",
})

# --- Paths ---
WORKING_DIR = "D:/StudyRelated/Machine Learning Projects/NBA/dataset"

IMAGE_DIR_PATH = os.path.join(WORKING_DIR, "images")
SAMPLE_DIR_PATH = os.path.join(WORKING_DIR, "sample_images")
SAMPLE_OUTPUT_DIR_PATH = os.path.join(WORKING_DIR, "sample_annotation")
FULL_OUTPUT_DIR_PATH = os.path.join(WORKING_DIR, "autoAnnotation")

# --- Thresholds ---
BOX_THRESHOLD = 0.45   # stricter to reduce noise
TEXT_THRESHOLD = 0.3

def clear_folder(path):
    """Remove all files and subfolders in a folder."""
    if os.path.exists(path):
        for f in os.listdir(path):
            f_path = os.path.join(path, f)
            if os.path.isfile(f_path) or os.path.islink(f_path):
                os.remove(f_path)
            elif os.path.isdir(f_path):
                shutil.rmtree(f_path)
    else:
        os.makedirs(path, exist_ok=True)

# --- Step 1: Clear input/output folders for samples ---
clear_folder(SAMPLE_DIR_PATH)
clear_folder(SAMPLE_OUTPUT_DIR_PATH)

# --- Step 2: Collect random 10 images ---
all_images = [f for f in os.listdir(IMAGE_DIR_PATH) if f.endswith(".png")]
sample_images = random.sample(all_images, 10)

# --- Step 3: Copy sampled images into sample folder ---
for img in sample_images:
    shutil.copy(os.path.join(IMAGE_DIR_PATH, img),
                os.path.join(SAMPLE_DIR_PATH, img))

print(f"Copied {len(sample_images)} images into {SAMPLE_DIR_PATH}")

# --- Step 4: Initialize GroundingDINO ---
base_model = GroundingDINO(
    ontology=ontology,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD
)

# --- Step 5a: Run annotation on sampled images ---
dataset = base_model.label(
    input_folder=SAMPLE_DIR_PATH,
    extension=".png",
    output_folder=SAMPLE_OUTPUT_DIR_PATH
)

print("✅ Sample annotation complete. Results saved to:", SAMPLE_OUTPUT_DIR_PATH)

# --- Step 5b: (OPTIONAL) Run annotation on ALL images ---
# Uncomment this once you are happy with the sample results
#"""
clear_folder(FULL_OUTPUT_DIR_PATH)
full_dataset = base_model.label(
    input_folder=IMAGE_DIR_PATH,
    extension=".png",
    output_folder=FULL_OUTPUT_DIR_PATH
)
print("✅ Full annotation complete. Results saved to:", FULL_OUTPUT_DIR_PATH)
#"""
