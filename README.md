# High-altitude tiny object detection with RT-DETR-L, ESRGAN, SwinIR, SAHI and NWD loss

Task-driven super-resolution for tiny aerial object detection on VisDrone2019-DET
and AI-TOD. Work from my DRDO internship.

## Overview

The goal was to detect very small targets, such as vehicles and aircraft, in
imagery taken from around 6 km up. At that altitude an object covers only a few
pixels and is blurred by the atmosphere and limited by sensor resolution.

The project combines:

* RT-DETR-L, a transformer-based detector
* ESRGAN and SwinIR for super-resolution (SR)
* SAHI for slicing-assisted inference
* NWD (Normalized Wasserstein Distance) loss for localizing tiny objects
* Joint training of the SR and detection models
* A comparison against IoU-based localization

## The problem

Tiny objects are hard for standard detectors. Vehicles and aircraft cover very
few pixels, atmospheric blur and sensor noise hide them further, and a small
localization error changes IoU a lot, which makes IoU-based losses unstable for
very small boxes.

The questions I set out to test:

* Does super-resolution make tiny objects easier to detect?
* Does SR trained for detection (task-driven) beat SR trained to look good?
* Does NWD loss make localization of very small targets more stable?
* Do transformer-based SR models beat GAN-based SR on aerial imagery?

The results so far are in [What the notebooks measured](#what-the-notebooks-measured);
most of these are still open.

## Datasets

### VisDrone2019-DET

Used for extracting tiny vehicles, high-altitude vehicle detection, the blur
and noise degradation experiments, SR-assisted detection, and RT-DETR-L
training and evaluation. The filtered classes are cars, vans, trucks and buses.

### AI-TOD

AI-TOD is built for tiny object detection in aerial imagery. Objects are often
5–15 pixels wide and sparsely distributed, so scale is a problem and
localization errors matter a lot.

Used for ESRGAN/SwinIR joint training, NWD-based localization, tiny-object
benchmarks and the IoU vs NWD comparison.

## Pipeline

```text
VisDrone / AI-TOD
        ↓
Tiny Vehicle Filtering
        ↓
Rotation Augmentation
        ↓
Blur + Noise Degradation
        ↓
Super-Resolution
(ESRGAN / SwinIR)
        ↓
SAHI Slicing
        ↓
RT-DETR-L + NWD Loss
        ↓
Tiny Object Detection
```

## Components

### RT-DETR-L

The main detector. I picked it because it's an end-to-end transformer detector
with anchor-free localization and global feature modelling, it adapts well to
small objects, and it runs in real time.

### ESRGAN

Enhanced Super-Resolution GAN, used for 2× and 4× SR and for joint SR-detection
training. It produces sharp textures and realistic-looking images, but at high
magnification it hallucinates textures, can make very small targets
semantically inconsistent, and made classification unstable when SR was
aggressive.

### SwinIR

A transformer-based SR model, explored as an alternative to ESRGAN. The reason
for trying it: self-attention should keep global structure more consistent,
hallucinate less texture and give RT-DETR more stable features. If that holds,
it should improve precision, semantic consistency, the stability of 4× SR
training and AP_small. None of this has been measured yet; see
[Hypotheses still to test](#hypotheses-still-to-test).

### SAHI (Slicing Aided Hyper Inference)

Tiny objects are very hard to find in a full-resolution aerial image. SAHI cuts
the image into overlapping slices, runs detection on each tile and merges the
predictions:

```text
Large Image
     ↓
Overlapping Slices
     ↓
Detection on Individual Tiles
     ↓
Prediction Merging
```

Each tiny object becomes larger relative to the detector input, which is meant
to improve recall and small-object localization.

## NWD loss (Normalized Wasserstein Distance)

A main part of this project is comparing IoU-based and NWD-based localization.

### The problem with IoU for tiny objects

For very small objects, a 1–2 pixel shift can reduce IoU drastically, so small
localization errors give unstable gradients and learning becomes inconsistent.

```text
Ground Truth Box: 6×6
Prediction Shifted by 2 Pixels
→ IoU collapses rapidly
```

### How NWD handles it

NWD models a bounding box `(cx, cy, w, h)` as a Gaussian `N(μ, Σ)` instead of
a rectangle, where:

```text
Σ = [[w²/12, 0],
     [0, h²/12]]
```

Compared with IoU, this should give smoother gradients, more stable
localization learning, less sensitivity to small pixel shifts, and better
alignment and optimization for tiny targets. It should matter most on AI-TOD,
tiny VisDrone vehicles and high-altitude ISR imagery. These benefits come from
the NWD paper; this repo hasn't confirmed them yet.

## Loss functions

* Pixel loss keeps spatial consistency, geometric structure and an accurate
  reconstruction.
* VGG-based perceptual loss keeps semantic structure, makes objects
  recognizable and keeps features consistent.
* Adversarial (GAN) loss improves visual sharpness and perceptual detail, but
  can hallucinate textures on very small objects.

The joint training loss:

```text
L_total = L_cls + L_nwd + λL_sr
```

where `L_cls` is the classification loss, `L_nwd` the NWD localization loss and
`L_sr` the super-resolution reconstruction loss.

## Joint training

The main idea was task-driven SR. Instead of training the SR model and the
detector separately, the pipeline trains them together:

```text
ESRGAN / SwinIR
        +
RT-DETR-L
```

That way the detector's gradients shape the SR reconstruction, so SR optimizes
for detectability instead of appearance and reconstructs with localization in
mind.

## Experiments

* High-altitude tiny object detection: VisDrone2019-DET, RT-DETR-L, ESRGAN and
  SAHI slicing-assisted inference.
* High-altitude vehicle detection: the filtered VisDrone tiny-vehicle set,
  RT-DETR-L, ESRGAN and SAHI inference.
* NWD vs IoU: AI-TOD and VisDrone with RT-DETR-L, ESRGAN and joint training,
  evaluated on localization stability, recall, precision and AP_small.
* Video: a high-altitude video detection pipeline with SAHI-assisted inference,
  an NWD vs IoU comparison and tiny-object tracking experiments.
* Cross-model comparison of YOLOv12-L, YOLOv26-L and RT-DETR-L, combined with
  ESRGAN, SwinIR (referenced in code; no recorded result), SAHI slicing and NWD
  localization.

## Observations from the joint training notebook

Read from the training logs of the joint ESRGAN + RT-DETR + NWD notebook
(`rtdetr-esrgan-nwd-joint-training-on-aitod-and-vis`):

* VisDrone 2× trained for 10 epochs (best loss 0.805).
* VisDrone 4× ran 3 epochs, not 10 (best loss 1.195).
* AI-TOD 2× logged all 10 epochs, but the output stops before the run's
  completion line.
* AI-TOD 4× has no training output in the notebook.

The final evaluation never ran on a trained model. The notebook reports
"Checkpoint not found" for all four runs (`runs/*/best.pt` missing), then prints
a table of 0.0000 for mAP@50, mAP@50-95, precision and recall. Those zeros are
fallback values, not measured performance, so these runs show nothing about
detection quality either way. Training logged "Best saved", but the checkpoints
were not found when evaluation ran; the cause isn't recorded in the notebook.

## Visualizations

The notebooks and reports include loss curves, NWD trends, SR reconstruction
comparisons, joint training summaries, qualitative ESRGAN/SwinIR outputs,
detection results and video inference output (`output_detected.mp4`).

## Tools

Python, PyTorch, OpenCV, Ultralytics RT-DETR, SAHI, ESRGAN, SwinIR, NumPy and
Matplotlib, run on Kaggle, Google Colab and Lightning AI.

## Hardware

| GPU         | Usage                             |
| ----------- | --------------------------------- |
| NVIDIA T4   | Baseline training                 |
| NVIDIA L4   | Optimized inference               |
| NVIDIA A10G | Large-scale SR + RT-DETR training |

## What's in this repository

```text
AI_TOD_dataset_pruning.ipynb                            AI-TOD filtering
High_alt_Vehicle_Detection.ipynb                        high-altitude vehicle detection
high-alt-tiny-obj-dtct-with-esrgan-full-pipeline.ipynb  ESRGAN on degraded frames
rtdetr-esrgan-nwd-joint-training-on-aitod-and-vis.ipynb joint ESRGAN + RT-DETR + NWD
yolo8-l-esrgan-joint-training-on-aitod-and-vis.ipynb    YOLOv8-L joint training
rtdetr-on-a-video.ipynb                                 RT-DETR on video
yolo-on-a-video-iou-nwd-comparision.ipynb               IoU-NMS vs NWD-NMS on video
output_detected.mp4                                     annotated video output
*.pdf, *.docx                                           reports for each experiment
```

Datasets, checkpoints and SR outputs aren't in the repo; the notebooks download
or generate them on Kaggle and Colab.

## What the notebooks measured

| Experiment | Result recorded in the notebook |
|---|---|
| ESRGAN on 8×-zoom degraded frames (`high-alt-tiny-obj-dtct-with-esrgan-full-pipeline`) | mAP@50 0.0588 clean → 0.0601 with ESRGAN (+0.0013); blur+noise 0.0503 |
| Joint ESRGAN + RT-DETR + NWD loss (4 runs) | Not evaluated: checkpoints missing, table shows fallback zeros (see above) |
| YOLO with IoU-NMS vs NWD-NMS on video (`yolo-on-a-video-iou-nwd-comparision`) | IoU: 12.6–13.7 detections/frame; NWD: 1.0/frame. Same speed (±0.3 ms) |
| YOLOv8-L on bicubic-upscaled inputs (ESRGAN step still a placeholder) | mAP@50 ≈ 0.55 in the final logged metrics |

What this supports: ESRGAN made almost no difference to mAP on the degraded
frames, and NWD-based NMS as implemented collapses detections to about one per
frame, which points to a threshold or scaling problem rather than an
improvement over IoU.

## Hypotheses still to test

* NWD for tiny-object localization. The motivation (smoother gradients for
  boxes a few pixels wide) is from the NWD paper. It is not shown here: the
  joint NWD runs above were never evaluated, and NWD-NMS suppressed almost all
  detections.
* Task-driven vs. visually optimized SR. Not compared directly yet.
* SwinIR vs. ESRGAN. SwinIR appears in the code, but no SwinIR result is
  recorded.
* Recall as the main failure mode. Plausible for tiny objects, but there is
  no error breakdown (missed vs. misclassified) in the notebooks yet.

## Next steps

* SwinIR + RT-DETR-L joint training
* Real-ESRGAN integration
* Detection-aware perceptual losses
* Dynamic SR scaling
* Adaptive NWD weighting
* Transformer-based task-driven SR
* Optimizing for AP_small
* Real-world ISR deployment experiments

## Objective

Improve high-altitude tiny-object detection with task-driven super-resolution,
transformer-based detection, slicing-assisted inference and localization-aware
training, for aerial surveillance and ISR.

## Citation

```bibtex
@misc{tiny_object_sr_rtdetr,
  title={Task-Driven Super-Resolution for Tiny Object Detection using ESRGAN/SwinIR, RT-DETR-L, SAHI, and NWD Loss},
  author={Arnav Deshpande},
  year={2026}
}
```

## Author

Arnav Deshpande, Indian Institute of Technology Indore. I work on machine
learning, computer vision, tiny object detection, aerial AI systems,
transformer-based detection and super-resolution.
