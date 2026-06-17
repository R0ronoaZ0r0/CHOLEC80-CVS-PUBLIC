# Cholec80-CSV Preprocessing Pipeline

This document summarizes how this repository turns the raw Cholec80-CVS videos and surgeon annotations into the frame-level CSV dataset used by Colenet.

## 1. Raw Inputs

The pipeline starts from:

- Cholec80 videos in `data/cholec80/videos/`
- Phase annotations in `data/cholec80/phase_annotations/`
- Surgeon CVS annotations in `data/surgeons_annotations.xlsx`
- Configuration in `config/config.json`

The original videos are 25 FPS.

## 2. Find Valid Video Range

Script:

```bash
data_preprocessing/get_valid_frames.py
```

This script reads each phase annotation file and finds the first frame labeled `ClippingCutting`.

It stores the frame before that phase as the last valid frame:

```text
final_frame = first ClippingCutting frame - 1
```

Output:

```text
results/videos_frames_index.csv
```

## 3. Extract Frames From Videos

Script:

```bash
data_preprocessing/video_2_frames.py
```

This script reads each video and saves every frame from frame `0` through the video's `final_frame`.

At this stage, frames are still extracted at the original video rate:

```text
25 FPS
```

Each saved frame is cropped to remove the black border/mask around the laparoscopic image. The crop is computed from pixels brighter than a small threshold:

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
black_pixels = np.column_stack(np.where(gray > 10))
new_frame = frame[y_min:y_max, x_min:x_max]
```

Output structure:

```text
data/cholec80_frames/
  video01/
    0.jpg
    1.jpg
    ...
  video02/
    0.jpg
    1.jpg
    ...
```

## 4. Convert Surgeon Annotations To Frame Labels

Script:

```bash
data_preprocessing/annotations_2_labels.py
```

This script reads:

```text
data/surgeons_annotations.xlsx
```

For each video, it creates frame-level labels for the three CVS criteria:

- `two_structures_score`
- `cystic_plate_score`
- `hc_triangle_score`

Important constants:

```python
truncation_ratio = 0.85
video_fps = 25
target_fps = 5
```

The script first creates labels for every valid frame. Then it truncates the video labels:

```text
initial_frame = floor(total_valid_frames * 0.85 / 25) * 25
```

That means it keeps the later part of the valid video region and removes a large early section with many zero labels.

Then it downsamples from 25 FPS to 5 FPS:

```text
For each 25-frame block, randomly choose 5 frames.
```

So the CSV dataset is effectively:

```text
5 FPS
```

Output structure:

```text
data/surgeons_annotations/
  video01.csv
  video02.csv
  ...
```

Each CSV row points to one selected frame and its labels.

## 5. Create Train, Validation, And Test CSVs

Script:

```bash
data_preprocessing/get_training_sets.py
```

This script splits the data by video, not by individual frames.

The split is:

```text
train: 50 videos
val:   15 videos
test:  15 videos
```

Outputs:

```text
data/train.csv
data/val.csv
data/test.csv
data/stats.csv
```

In the current generated dataset:

```text
train: 43,850 frame rows
val:    9,615 frame rows
test:   9,295 frame rows
total: 62,760 frame rows
```

There are 62,752 unique image files referenced by these CSVs because a few rows are duplicates from random sampling with replacement in short chunks.

## 6. Dataset Loading

Class:

```python
colenet/cholec80csv_dataset.py
```

`Cholec80CSVDataset` reads one of:

```text
data/train.csv
data/val.csv
data/test.csv
```

For each row, it loads:

```text
<root_dir>/<video_name>/<image>.jpg
```

The raw CSV labels can contain `0`, `1`, or `2`, but the dataset converts them into binary labels:

```python
score = int(min(1, row[score_column]))
```

So:

```text
0 -> 0
1 -> 1
2 -> 1
```

The model is trained to answer whether each CVS criterion is present or not.

## 7. Training Image Transforms

Training transforms in `colenet/trainer.py`:

```python
transforms.ToPILImage()
transforms.Resize(256)
transforms.CenterCrop(224)
transforms.RandomHorizontalFlip()
transforms.ToTensor()
transforms.Normalize([0.485, 0.456, 0.406],
                     [0.229, 0.224, 0.225])
```

The only random image augmentation during training is:

```text
RandomHorizontalFlip
```

Resize, center crop, tensor conversion, and ImageNet normalization are deterministic preprocessing steps.

## 8. Validation And Test Image Transforms

Validation and test transforms are deterministic:

```python
transforms.ToPILImage()
transforms.Resize(256)
transforms.CenterCrop(224)
transforms.ToTensor()
transforms.Normalize([0.485, 0.456, 0.406],
                     [0.229, 0.224, 0.225])
```

There is no random augmentation for validation or test.

## 9. Model Training Summary

With `backbone = "resnet"`, Colenet uses:

```text
ImageNet pretrained ResNet50
```

and replaces the final classification layer with a 3-output linear layer:

```python
model.fc = nn.Linear(model.fc.in_features, 3)
```

The three outputs correspond to:

- `two_structures_score`
- `cystic_plate_score`
- `hc_triangle_score`

Training uses multi-label binary classification:

```python
nn.BCEWithLogitsLoss(pos_weight=...)
```

During evaluation, logits are passed through sigmoid and thresholded at `0.5`:

```text
sigmoid(logit) >= 0.5 -> positive
sigmoid(logit) < 0.5  -> negative
```

## Short Version

```text
Raw 25 FPS videos
 -> find final valid frame before ClippingCutting
 -> extract valid frames at 25 FPS
 -> crop black laparoscopic border
 -> convert surgeon annotations to frame labels
 -> keep later valid-video region
 -> randomly sample 5 frames from every 25-frame second
 -> split by video into train/val/test
 -> load images with resize, crop, normalize
 -> train Colenet as a 3-label binary CVS classifier
```
