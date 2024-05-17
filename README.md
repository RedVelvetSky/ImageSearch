# Project Title: Image Retrieval Using CLIP Model

![GUI](./img.png)

## Overview
This project is an implementation of an image retrieval system using the CLIP (Contrastive Language–Image Pretraining) model. The system allows users to search for images using text queries and provides a graphical user interface (GUI) for displaying and interacting with the results.

## Features
- Load and display a large dataset of images.
- Search for images using text queries based on the CLIP model.
- Select and deselect images.
- Update the display of images based on various similarity measures.
- Zoom in on images for a closer view.
- Save and load image features.

## Requirements
- Python 3.7+
- Required Python libraries: `faiss`, `numpy`, `torch`, `Pillow`, `pandas`, `requests`, `scikit-learn`, `open_clip_pytorch`, `tkinter`

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required libraries:
    ```sh
    pip install -r requirements.txt
    ```

4. Set the environment variable:
    ```sh
    export KMP_DUPLICATE_LIB_OK=TRUE  # On Windows: set KMP_DUPLICATE_LIB_OK=TRUE
    ```

## Dataset

Place your image dataset in the `dataset_path` directory. Ensure that the images are in `.png`, `.jpg`, or `.jpeg` format.

## Usage

Run the main script to start the GUI:

```sh
python main.py
```

## Main Components

### Model

The `Model` class is responsible for loading the CLIP model and features, as well as performing text-based image searches.

### GUI

The GUI is built using Tkinter and provides the following functionalities:
- Display images in a grid format.
- Search for images using text queries.
- Select and deselect images by clicking on them.
- Zoom in on images by right-clicking on them.
- Update the displayed images based on different similarity measures (Cosine, HNSW, IVF, Inner Product).

### Image Processing

The image processing functionalities include:
- Encoding images in batches and saving their features.
- Loading precomputed features from a file.
- Scaling features based on importance scores.
- Refining search results based on positive and negative image selections.

### Utility Functions

The project includes various utility functions to support the main functionalities:
- `update_negative_indices`: Updates the list of negative indices based on user selections.
- `setup_faiss_index`: Sets up a FAISS index for efficient similarity search.
- `refine_search_v3`: Refines search results using positive and negative image selections.
- `iterative_search`: Performs iterative search refining based on positive and negative indices.
- `compute_importance_scores`: Computes importance scores for feature dimensions based on positive and negative selections.
- `scale_features_by_importance`: Scales features by their importance scores.
- `weighted_centroid`: Calculates a weighted centroid of the features.

## Example Usage

To encode images and save their features:

```sh
from features_extractor_compressed import encode_images_in_batches

image_folder = "path/to/your/image/folder"
output_file = "features.npy"
encode_images_in_batches(image_folder, batch_size=32, output_file=output_file)
```

To load the saved features:

```sh
from features_extractor_compressed import load_features

features_tensor = load_features("path/to/features.npy")
print(features_tensor.shape)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The CLIP model by OpenAI.
- The FAISS library for efficient similarity search.
- The OpenAI API for providing pretrained models.

## Contact

For any questions or suggestions, please contact [your_email@example.com](mailto:your_email@example.com).
