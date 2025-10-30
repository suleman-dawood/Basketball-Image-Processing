Link to drive for dataset: https://drive.google.com/file/d/1e5QqS9OcaIivUd8F8C1z65qqxypKzMAc/view?usp=sharing

STEP 1: Clone Repo

STEP 2: Download autoAnnotations zip from above

STEP 3: Put the folder inside zip that is called autoAnnotations inside the dataset folder (same level as "yolo_training"). You can delete the vis_color_clean folders if you want.

STEP 4: Go through all 5 Python scripts and change all the file paths inside them so that they are suitable for you, good luck.

STEP 5: recreate my virtual environemnt with the following bash commands:

python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

pip install -r requirements.txt


STEP 6: Run script 1. Should generate all frames into dataset/images

STEP 7: Open the folder in file explorer and manually delete irrelevant frames (olympics graphics at the end)

STEP 8: Run script 2. This'll take like 5 hours on a RTX 3070. Does auto annotation using autodistill and grounding dino.

STEP 9: Run script 3. Postprocessing script to address some issues when making predictions with script 2.

STEP 10: Run script 4. Training the YOLO model. Set at 50 epochs took about 5 hours on a 3070 ~~10 epochs/hr

STEP 11: Run script 5 to test the model on a video of your choice. It will create an MP4 which you can watch to see results.



NOTES: There is some logic in there for doing sample annotation and sample training. If teammate and you wanna understand how it works msg the discord. 