# AnySkin Slip Detection

Slip detection from tactile sensor data using USDConvNetLite, a residual CNN with
about 5k parameters. The model classifies short windows of sensor readings as slip or
no-slip and is small enough for real-time use on embedded hardware.

The method is based on Zhao et al., [Universal slip detection of robotic hand with
tactile sensing](https://pmc.ncbi.nlm.nih.gov/articles/PMC11843555/) (Frontiers in
Neurorobotics, 2025), and is applied to the Pollen Robotics
[anyskin_slip_detection](https://huggingface.co/datasets/pollen-robotics/anyskin_slip_detection)
dataset.

## Dataset

The data was recorded with a Reachy 2 gripper and an AnySkin magnetic tactile sensor
(5 magnetometers with x, y and z readings, giving 15 channels). Each of 16 objects was
grasped and pulled in different directions until it slipped. A separate recording
contains no slip.

| | |
|---|---|
| Channels | 15 |
| Objects | 16, plus one no-slip recording |
| Rows | 127,445 (about 10 minutes at ~230 Hz) |
| Slip events | 64 (4 per object) |
| Slip rows | 14.5% |

The dataset is small and imbalanced. Because neighbouring windows overlap, the number
of independent examples is closer to the 64 slip events than to the number of rows.

## Method

The sensor stream is divided into overlapping windows of 16 readings. Each window is
labelled with the slip state of its last reading.

Four input configurations are compared, following the paper:

| Configuration | Input | Shape |
|---|---|---|
| Base | Raw readings | 16 x 15 |
| FFT | FFT magnitudes stacked on raw readings | 32 x 15 |
| Filter | High-pass filtered readings | 16 x 15 |
| FFT + Filter | FFT magnitudes stacked on filtered readings | 32 x 15 |

USDConvNetLite consists of a convolutional layer, 4 residual blocks, adaptive average
pooling and a linear classifier.

Two evaluation protocols are used:

- **Random 80/20 split.** Overlapping windows from the same recording appear in both
  the training and test sets, so this score is optimistic.
- **Leave-one-object-out.** The model is trained on all other objects and tested on
  the held-out object, repeated for each of the 16 objects. This measures performance
  on objects not seen during training.

## Results

Leave-one-object-out results, averaged over 16 objects:

| Configuration | Accuracy | Slip precision | Slip recall |
|---|---|---|---|
| Base | 82.42 ± 5.44% | 0.52 | 0.30 |
| FFT | 82.45 ± 5.30% | 0.52 | 0.36 |
| Filter | 81.93 ± 5.98% | 0.50 | 0.32 |
| FFT + Filter | 83.18 ± 4.42% | 0.55 | 0.34 |

FFT + Filter has the highest mean accuracy and the lowest standard deviation across
objects.

Accuracy alone overstates performance. 82.4% of test windows are no-slip, so a model
that always predicts no-slip would reach a similar accuracy. Slip recall is between
0.30 and 0.36, meaning roughly two thirds of slip windows are missed. Improving slip
precision and recall is the next stage of the project.

## Project structure

```
src/
  data.py                data loading, windowing, high-pass filter, FFT
  model.py               USDConvNetLite
  train.py               training with random-split and leave-one-object-out evaluation
  compare_configs.py     runs the four input configurations
  build_table.py         results table
  plot_curves.py         train and test accuracy per epoch
  plot_confusion.py      confusion matrices
  class_distribution.py  slip and no-slip counts per object
  evaluate.py            evaluation of a saved model (precision, recall, F1)
```

## Usage

```bash
pip install -r requirements.txt
```

Scripts are run from the project root. The `--from_hub` flag downloads the dataset
from Hugging Face.

```bash
# Train a model (saved to checkpoints/model.pt)
python src/train.py --from_hub

# Compare the four configurations and generate the table and plots
python src/compare_configs.py --from_hub
python src/build_table.py
python src/plot_curves.py
python src/plot_confusion.py

# Slip and no-slip counts per object
python src/class_distribution.py

# Evaluate a saved model
python src/evaluate.py --from_hub --checkpoint checkpoints/model.pt
```

Outputs are saved to `results/`.

## Future work

- Address class imbalance with a weighted loss or oversampling of slip windows
- Select the best epoch by F1 on a validation set instead of test accuracy
- Tune the decision threshold for the slip class
- Report event-level metrics: slips detected out of 64, detection delay and false
  alarm rate
- Evaluate longer windows. At ~230 Hz, 16 readings cover about 70 ms, compared with
  about 300 ms in the paper
