# PLM ML Twins Classification

A machine learning application for twin boundary classification and grain segmentation in metallic microstructures, built around a Tkinter GUI and YOLOv8-based segmentation models.

## Project Overview

This project analyses Polarised Light Microscopy (PLM) images of metallic alloys to:

- **Segment grains** from orientation/pseudocolour images
- **Detect and classify twin boundaries** (Tension vs. Compression) using deep learning
- **Generate pseudocolour images** from stacks of grayscale orientation images, as a substitute for EBSD-style orientation mapping

The published methodology behind this approach is described in the paper below.

## Publication

For the full methodology and validation of this technique, see:

[https://www.sciencedirect.com/science/article/pii/S2590049826001566](https://www.sciencedirect.com/science/article/pii/S2590049826001566)

## Features

- **Interactive GUI** (Tkinter) for loading images, generating pseudocolour maps, and running segmentation/classification
- **Resolution-aware processing** — the GUI requires the image resolution (µm/px) before unlocking segmentation, and warns if the resolution falls outside the model's training range
- **Two processing modes**:
  - *Simple Segmentation* — fast grain boundary detection
  - *Twins Classification* — full twin detection and classification, combined with a PLM orientation map
- **YOLOv8-based models** for segmentation and classification
- **Side-by-side visualisation** of segmentation results and PLM maps

## Installation
To set up the project on your local machine, follow these steps:

1. Clone the repository:
  ```
   git clone https://github.com/girerdth/PLM_ML_Twins_Classification.git
   cd PLM_ML_Twins_Classification
  ```

2. Install the necessary dependencies:
```
  pip install -r requirements.txt
```
## Usage
### Running the Application
Execute the main script to launch the GUI:
```
python Twins_Classification.py
```
### GUI buttons

| Button | Description |
|---|---|
| **Select Image** | Load an existing pseudocolour image (PNG/JPG/JPEG) |
| **Create Pseudocolour** | Generate a pseudocolour image from grayscale orientation images |
| **Simple Segmentation** | Run basic grain boundary detection |
| **Twins Classification** | Run full twin detection/classification with PLM mapping |

### Workflow

1. **Load or generate an image**
   Click **Select Image** to load a pseudocolour image directly, or click **Create Pseudocolour** and choose three grayscale orientation images (0°, 40°, and 80° are recommended) to build one.

2. **Enter the image resolution**
   Once an image is loaded, a resolution panel appears below the toolbar. Enter the resolution in **µm/px** and click **Confirm**.
   - The GUI resizes the image to 640 px wide internally (matching the model's input size) and recalculates the equivalent resolution at that size.
   - If the recalculated resolution is **at or below the model's threshold (0.39 µm/px)**, processing is unlocked with a confirmation message.
   - If it's **above the threshold**, processing is still unlocked, but the GUI warns that the model was not trained at this resolution and results may be unreliable.
   - Editing the value after confirming re-locks processing until you confirm again.

3. **Run processing**
   - **Simple Segmentation** for grain boundaries only.
   - **Twins Classification** for full twin detection — you'll be prompted to select the folder containing the 18 (or 36) grayscale orientation images for that sample, used to estimate crystallographic orientation. Results appear in the side-by-side display panels.

### Example: full Twins Classification workflow

1. Click **Create Pseudocolour**, select the three grayscale images, and save the resulting pseudocolour image.

   ![step1](files/media/step1.png)

2. Enter the resolution (µm/px). The GUI checks it against the model's training threshold after resizing to 640 px.

   ![step2](files/media/step2.png)

3. Select the orientation folder and view the final classification result.

   ![step3](files/media/step3.png)

### Project Structure
```Code
PLM_ML_Twins_Classification/
├── Twins_Classification.py    # Main GUI application
├── Terminal_method.py         # Terminal-based processing methods
├── requirements.txt           # Python dependencies
├── source_code/
│   ├── pseudoimage.py         # Pseudocolour image generation
│   ├── run_models.py          # ML model execution (simplify & amplify methods)
│   ├── Grain_functions.py     # Grain/twin geometry, overlap resolution, contour utilities
│   └── Grain_Orientation.py   # Orientation colour-mapping
├── data/                      # Input data directory
├── models/                    # Pre-trained ML models
├── files/                     # Processing output files and media
├── segmentation_results/      # Saved segmentation/classification outputs
└── .idea/ 
```
## Key dependencies

**Core**
- `tkinter` — GUI (built-in)
- `opencv-python` (`cv2`) — image processing
- `Pillow` — image display in GUI
- `numpy`, `scipy` — numerical computing

**Machine learning**
- `torch` — deep learning framework
- `ultralytics` — YOLOv8 segmentation/classification models
- `scikit-image` — advanced image processing (skeletonisation, contours, regionprops)

**Materials science / geometry**
- `shapely` — polygon geometry for grain/twin overlap analysis
- `skan` — skeleton graph analysis (branch decomposition)
- `orix`, `diffpy.structure` — crystallographic orientation analysis (where applicable)

See `requirements.txt` for exact pinned versions, including:

| Package | Version |
|---|---|
| Python | ≥ 3.8 |
| PyTorch | 2.3.1 |
| Ultralytics | 8.2.51 |
| OpenCV | 4.11.0.86 |
| NumPy | 1.26.4 |
| Pandas | 2.3.1 |
| Matplotlib | 3.10.3 |
| scikit-image | 0.25.2 |

## Technical details

### Main components

- **`App` (Twins_Classification.py)** — manages the GUI: image loading/display, the resolution confirmation panel, and triggering segmentation/classification.
- **`pseudoimage` module** — builds pseudocolour images from orientation-stack grayscale images, with contrast/CLAHE enhancement.
- **`run_models` module** — wraps YOLOv8 inference:
  - `simplify_method` — basic segmentation
  - `amplify_method` — full twins classification
- **`Grain_functions` / orientation pipeline** — grain extraction from skeletons, twin/grain overlap resolution, neighbour-finding, misorientation-angle calculations, and twin-type classification (Tension/Compression).

### Image processing pipeline

1. Load (or generate) the pseudocolour image
2. Confirm image resolution (µm/px); check against the model's training threshold at 640 px width
3. Run the segmentation or classification model
4. Extract grains/twins from the model output, resolve overlaps, and estimate crystallographic orientation from the orientation image stack
5. Classify twin type by misorientation angle relative to identified parent grains
6. Overlay contours on the pseudocolour image and display alongside the PLM orientation map

## Author

**Thomas Girerd**
Created: February 10, 2026

