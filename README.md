# PointObject

Pipeline running in IPython notebooks for analysing point localisation data

### Features ###
* Binning of localisation data into movie frames
* Clustering using DBSCAN (manual cluster selection and optional cluster refinement)
* Contour finding using Kernel Density Estimation and morphological contour fitting
* Curvature calculation

### System requirements ###

* Python3 with matplotlib, numpy, scipy, sklearn, etc. (for example [Anaconda Python](http://continuum.io/downloads))
* Qt for python (might be optional)
* IPython Notebooks
* Please note that generating .mp4 movies is only possible if the external ffmpeg binaries are available

### Usage ###
* For general usage please see the example files in the ```examples``` folder
* For a brief FAQ section please refer to the [wiki](https://github.com/nberliner/PointObject/wiki)

### Installation ###
* Install [Anaconda Python](http://continuum.io/downloads) (Python3)
* Update the python installation by running

    ```
    $ conda update conda
    $ conda update anaconda
    ```
    in an Anaconda command prompt.
* Get the source code from GitHub
    You need to install git if it is not already available. On Windows you can use [GitHub for Windows](https://windows.github.com/) and use the Git Shell.
    The source code will be placed in a folder called ```PointObject``` in the current directory of the command prompt. Before cloning into the GitHub repository we will thus change to a user specified directory (```userDirectory```).
    
    ```
    cd userDirectory
    git clone https://github.com/nberliner/PointObject.git
    ```
    This will create a folder ```PointObject``` in ```userDirectory```.
* Put the [ffmpeg](http://ffmpeg.org/download.html) binaries into the folder

  `userDirectory/PointObject/external/ffmpeg/win64`
  
  (replace ```win64``` with `linux` if you are working on a linux machine). Adjust the `_ffmpegLinux` or `_ffmpegWin` variable in `lib/movieMaker.py` according to your version.
* [Optional] You can create a file encapsulating the commands necessary to start the IPython Notebook server. On a Windows machine create a file ```startPointObject.cmd``` in your ```userDirectory/PointObject``` folder and copy/paste the following line
    ```
    ipython notebook --notebook-dir=notebooks
    ```
    You should now be able to run the IPython Notebooks by double clicking the file.

* [Optional] If you are sharing data, notebooks etc. via the switch.drive.ch owncloud instance of EPFL you can create links to the ownCloud folder. This will enable you to use relative paths for importing data which makes it easier to share notebooks and to work together on the analysis. There should be three folders in the owncloud folder, i.e. `data`, `external`, `notebooks`. By linking these folders you will firstly have the necessary ffmpeg binaries and secondly you can access the shared data and notebooks. On Windows use the `mklink` commmand from a DOS prompt.
    ```
    mklink /J C:\Path\To\Link\data C:\User\owncloud\PathToMito\data
    mklink /J C:\Path\To\Link\external C:\User\owncloud\PathToMito\external
    mklink /J C:\Path\To\Link\notebooks C:\User\owncloud\PathToMito\notebooks
   ```
