# Integrating live cell imaging with fixed cell barcoding
Notebooks and scripts for tracking and analyzing cells with a nuclear marker and other fluorescence channels. I would HIGHLY recommend you install this on a computer running Linux, since the deep learning models used in this pipeline do not work on Windows and Macs often don't have the correct binaries for installing them.

## Getting started

### Set up a Python virtual environment

Python virtual environments are a staple in coding with Python. Think of them as containers for your Python packages and code so that you can run your scripts. To set up a virtual environment, ensure you have Python installed on your computer. You can run `which python` to get the location of your Python file, and `python --version` or `python3 --version` to get the version number. We want to use Python version 3.10 or higher.

Once you have Python installed, do the following:

1. Pull up the command terminal and navigate to the folder where you'd like to do the analysis.
2. Clone this Github repo to the machine you will be using to do your analysis using   `git clone`.
3. Change your directory into the newly-created folder.
3. Create a virtual environment in this folder using  `python -m venv .venv` or `python3 -m venv .venv`. This will create a folder called `.venv` in your working directory that will house your Python packages (modules used for analysis).
4. Run `source .venv/bin/activate` to activate the virtual environment. This places you in the "container" that is the virtual environment, where you can install packages without chainging your normal workspace. To deactivate your virtual environment, simply run the command `deactivate`.
5. `pip` is the primary mode of installing packages in Python. Pip can be told to install multiple packages at once using a `requirements.txt` file, which lists packages and their versions. Run `pip install -r requirements.txt` to install required packages.

### DeepCell API Key

To use the DeepCell models, you first need to register for a DeepCell account and receive your API key. This key allows you to download the DeepCell models needed for segmenting and tracking. Visit the [DeepCell website](deepcell.org) and follow the instructions to generate the key.

Once you have the API key, edit your `.bashrc` or `.bash_profile` file in your home directory. At the end of the file, add the line `DEEPCELL_ACCESS_TOKEN=<generated API key>`. Exit and relaunch your command temrinal. This key is good for a set amount of time, so use it promptly.

## Workflow
Cells must have a nuclear marker for tracking (DAPI, H2B-HaloTag, H2B-mCherry, etc). In addition, the cells must have fluorescence in another channel. The images should be `.nd2` files, which are Nikon's native file format for images.

Parameters for live cell analyses are stored in a required `metadata.toml` files. An example of this can be found in `sample_wellmap/example.toml`. If you need to refer to the Wellmap package's syntax, visit their [documentation](wellmap.readthedocs.io).

Required parameters are as follows:

**Storage paths:**
```
path_to_live = 'data'
output_dir = 'data'
experiment_date = 20250429
```

`path_to_live` (`str`): directory to where raw data is stored. Can be relative to the home directory, or absolute relative to root (`/`).

`output_dir` (`str`): directory to where processed data is stored. Can be relative to the home directory, or absolute relative to root (`/`).

`experiment_date` (`int`): date of experiment for calling the `metadata.toml` file.

**Experiment channels and metadata:**
```
live_channels = ['EBFP', 'mClover3', 'mOrange2', 'mIRFP703']
registration_channel = 3
registration_type = 'local'
save_mask = true
treatment_frame = 4
live_nuc_channel = 3
num_sites = 1
nuc_area = 300
dilation_size = 15
```

`live_channels` (`list`): a list containing the names of each channel *in order*.

`registration_channel` (`int`): which channel you want to register the images on. Most often is the same as the nuclear marker, since this changes the least throughout the time lapse.

`registration_type` (`str`: `'local'` or `'global'`):
- `'local'`: each image is registered independently. This takes longer since the phase cross-correlation must be done for each image individually.
- `'global'`: all images on the same well plate should be subjected to the same jitter during the timelapse, which can be modeled using a median offset. All images are analyzed for individual offsets, then the median offset for each frame is calculated. This offset is applied to each image.

`save_mask` (`bool`): since nuclear segmentation takes the longest, setting this to `true` will save the nuclear masks for running the pipeline again.

`treatment_frame` (`int`): The frame at which the treatment is addded to the cells.

### Live-cell imaging analysis

#### Preprocessing
1. Images are converted from `.nd2` files, Nikon's native file format, to numpy arrays. 
2. Nuclei are segmented with `DynamicNuclearNet` from [DeepCell Applications](https://github.com/vanvalenlab/deepcell-applications). At each timepoint, the mask is then cleaned by removing nuclei smaller than the `nuc_area` variable. It is then saved to its own file in the output folder. This step takes the longest and, after tuning segmentation parameters, should only be done once.
3. Images are registered based on local jitter (`local` selection in `metadata.yml` file), or a median offset across all wells (`global` selection in `metadata.yml` file).
4. Registered images and segmented nuclei are then used to track single cells using `CellTracking` pipeline from [DeepCell Applications](https://github.com/vanvalenlab/deepcell-applications). 
5. Two files are written to your output folder specified in `output_dir`.
    1. A `_images.pickle` dictionary organized as follows:

        ```
        (dict)
            |
            |--- y (n_frames, rows, cols, 1) int32
            |
            |--- y_tracked (n_frames, rows, cols, 1) int32
            |
            |--- registered_image (n_frames, rows, cols, n_channels) uint16
            |
            |___ cyto_mask (n_frames, rows, cols, 1) int16
        ```
    2. A `_tracks.pickle` dictionary organized as follows:
        ```
        (dict)
            |
            |--- 1
            |     |--- label (int)
            |     |--- frames (list)
            |     |--- daughters (list)
            |     |--- capped (bool)
            |     |--- frame_div (int or None)
            |     |--- nuc_intensity (n_frames,) float
            |     |--- cyto_intensity (n_frames,) float
            |     |--- cell_intensity (n_frames,) float
            |     |--- exclude (bool)
            |     |___ channel_exclude (n_channels,) bool

            ... ...
            |
            |___ n_tracks
                |--- label (int)
                |--- frames (list)
                |--- daughters (list)
                |--- capped (bool)
                |--- frame_div (int or None)
                |--- nuc_intensity (n_frames,) float
                |--- cyto_intensity (n_frames,) float
                |--- cell_intensity (n_frames,) float
                |--- exclude (bool)
                |___ channel_exclude (n_channels,) bool
        ```

`nuc_intensity` (n_frames,) float: Mean nuclear intensity of every channel for that track

`cyto_intensity` (n_frames,) float: Mean intensity for every channel in the `cyto_mask` region of the image for that track.

`cell_intensity` (n_frames,) float: Mean intensity for every channel in the combined cyto and nuc regions for that track.

`exclude` (bool): Whether to exclude cell from further analysis.


After signal extraction, cell tracks can be plotted in a variety of ways. This wil depend on your goal for your experiment, so I would suggest creating your own Jupyter noteobok and making your own analysis scripts.

