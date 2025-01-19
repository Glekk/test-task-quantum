# Task2

## Contents
- [Solution Overview](#solution-overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)

## Solution Overview
This solution is aimed to solve Image Matching task for the Sentinel-2 satellite images.  
Is uses images from `https://www.kaggle.com/datasets/isaienkov/deforestation-in-ukraine` dataset as an example, but can be used with any other Sentinel-2 images.
The solution is based on the usage of one of the next algorithms:
- SIFT
- LoFTR
- DISK+LightGlue

Program takes two images as an input and returns stitched image with the matched keypoints.

## Project Structure 
- data folder - contains the images (.jpg included in reposiroty, .jp2 should be downloaded from the dataset or [here](https://drive.google.com/drive/folders/1L3JKSsLLVzUJQ9HUju9lRPbWa6wAX0uE?usp=sharing))
- notebooks folder - contains the notebooks with data oricessing and demonstration of inference
- src folder - contains the source code
    - algorithm.py - script containing realization of all algorithms
    - inference.py - script for inference
    - utils.py - script with utility functions
- .env - environment variables file
- requirements.txt - requirements file

## Prerequisites
- Python 3.10 (it was tested on this version)
- pip
- Jupiter Notebook\VS Code with extensions (to run the notebooks)

### Installation
1. Clone the repository
```
git clone https://github.com/Glekk/test-task-quantum.git
```
2. Navigate to the desired directory
```
cd test-task-quantum/Task2
```
3. Install the requirements
```
pip install -r requirements.txt
```

## Usage
- Before all you need to set up the environment variables in .env file, there is .env.example as an example.
- The data is already preprocessed and saved in the data folder.
- To run the inference you need go to the src folder and run the inference.py script. It has the next options:
```
usage: inference.py [-h] --alg {loftr,disk_lg,sift} --path0 PATH0 --path1
                    PATH1 [--img_shape IMG_SHAPE] [--no_equalize] [--clahe]
                    [--no_visualize] [--visualize_non_scaled_images] [--save]

Inference (using large images may take some time)

options:
  -h, --help            show this help message and exit
  --alg {loftr,disk_lg,sift}
                        Algorithm to use (loftr, disk_lg, sift)
  --path0 PATH0         Path to the first image
  --path1 PATH1         Path to the second image
  --img_shape IMG_SHAPE
                        Image shape as "height,width"
  --no_equalize         Do not equalize the images before processing
  --clahe               Apply CLAHE to the images before processing
  --no_visualize        Turn off visualization the matches
  --visualize_non_scaled_images
                        Visualize the matches with the original images
  --save                Save the visualization

```
- Example of usage:
```
cd src
python inference.py --alg loftr --path0 "..\data\jpg\T36UYA_20160212T084052_TCI.jpg"  --path1 "..\data\jpg\T36UYA_20190328T084011_TCI.jpg"
```
- To run the notebooks you just need to open them in appropriate environment (Jupiter Notebook\VS Code) and run all cells.
