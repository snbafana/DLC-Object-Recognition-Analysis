# Installation

First clone the repo:

Either use **anaconda**:

```sh
conda env create -f dlc-analy.yml
conda activate dlc-analy
```

Or install all the dependencies to a virtualenv with **pip**:

```sh
pip install -r requirements.txt
```

Then you can run user.py with

```sh
python user.py
```

Which will create all the necessary directories

# Usage

Before you use the analysis methods, you must have created a DLC project, extracted the training frames, labelled the training frames with at least the snout and neck. Then, you must also have trained the model in google colab, and processed all the object recognition videos to get csv outputs. Those csv outputs are necessary for the analysis. I have attached the colab that I used to this repo, 

**User.py** contains 10 different functions, each of which do different things. But here are some terms I use: 

- **Mask**: a numpy overlay with the shape of the video recording that contains the areas of the objects
- **NOR**: Novel Object Recognition
- **SLR**: Spontaneous Location Recognition

1. Quit the Program: Exit the application.
2. Create Masks: Allows the user to create masks from a video file.
3. Add Files to Masks: Enables the user to append specific files to existing mask files. Designating which mask goes with which csv input 
4. Run NOR Analysis of All Files in Masks: Performs Novel Object Recognition (NOR) analysis on all the files using the specified masks. 
5. Run NOR Analysis of One File with One Mask: Conducts NOR analysis on a single file using a specified mask.
6. Run SLR Analysis of All Files: Performs Spontaneous Location Recognition (SLR) analysis on all the files.
7. Run NOR with Only Centerpoint Values: Conducts NOR analysis using only the centerpoint values.
8. Run SLR with Centerpoints: Conducts SLR analysis using centerpoint values.
9. Visualize: Provides visualization using masks for selected video and corresponding CSV files.
10. Visualize Centerpoint: Similar to the visualize option but allows the user to select points in the video for visualization.


For Analysis types 4,5,6,9 - exploration is quantified by being a certain distance around the object and oriented toward the object. The parameters of orientation and distance can be modifed by the user

For Analysis types 7, 8 - exploration is quantified by just being a certain distance around the object. Parameters can also be modified here as well. 

The fileinfo.json holds all the utilities like, which mouse is in which cohort and some other key information. Here is a complete breakdown: 

- mice_in_cohorts: contains dictionaries of cohorts, which each have a separate demographic of mice. 
- mask_locs: where the masks are stored and which deeplabcut csv files they correspond to
- Threshold: for each of the cohorts, there exists an array where the first number delinates the first test in the testing phase (i.e. any test before was part of the familiarization phase). The second number was specifically for NOR when the novel object and familiar object were swapped, as it represents the first test number when that occured. 
- Between: the number of mice in a cohort. 

This code is not perfectly setup to transfer to new experiments, some tweaking must be done. I have commented the code so tweaks can be made (hopefully) easily. 

# Other Analysis

EXPLORE is as Deep Learning Convolutional Neural Network Solution for Object Recognition Paradigms.

Ibañez, V., Bohlen, L., Manuella, F. et al. EXPLORE: a novel deep learning-based analysis method for exploration behaviour in object recognition tests. Sci Rep 13, 4249 (2023). https://doi.org/10.1038/s41598-023-31094-w

https://github.com/Wahl-lab/EXPLORE

I used this method as an alternative for the DLC NOR & SLR pipeline, and I have also attached a colab that works with the labeled files from the EXPLORE pipeline. 

# Questions

If you have any questions, please contact me at snbafana@gmail.com

# Citations

