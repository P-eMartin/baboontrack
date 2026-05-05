# Packaging BaboonTrack

This is a package to perform detection, tracking and classification of Baboons.

## Description

TODO

## Dependencies

This package highly depends on [MegaDetector](https://megadetector.readthedocs.io/en/latest/index.html) which requires Python <3.14, >=3.9. Make sure to have the proper [python version](https://www.python.org/downloads/) installed before going to the next steps. You may replace the next commands `python` by `py -3.13` (for python 3.13) on Windows or `python3` or `python3.13` (when several python 3 are installed) for Linux and MacOS. 

<!-- **Side note:** The use of `pyenv` tool or similar to separate your python versions may not support the graphical user interface. -->

## Installation

If you downloaded this repository, make sure you to enter the following commands at the level of the package folder.

```
cd packaging_baboontrack
```

### Virtual environment (good practice)

Create a virtual environment. In this example the virtual environment will be stored in the hidden folder `.venv`.

```
python -m venv .venv
```

#### venv activation
**Linux and MacOS:**
```
source .venv/bin/activate
```
**Windows:**
```
.\.venv\Scripts\activate
```

### Package installation

This is a local wheel file proper to this repository. It is not yet published, hence publicly available, on pip.

**Linux and MacOS:**
```
pip install dist/packaging_baboontrack-0.0.1-py3-none-any.whl
```
**Windows:**
```
pip install .\dist\packaging_baboontrack-0.0.1-py3-none-any.whl
```

### ffmpeg (facultative)

You may [download ffmpeg](https://ffmpeg.org/download.html) for your distribution in order to create video outputs. 

<!-- ## Running physiotip

In python:

``` python
import physiotip
physiotip.run(gui=True)
```

For more information on possible options without graphical interface:

``` python
import physiotip
physiotip.run(help=True)
```

Run with provided images:

``` python
import physiotip
import os
ti_video = os.path.join('data','ti1')
physiotip.run(inputfolder=ti_video, gui=True)
``` -->

## Update packaging_baboontrack

This package is not published yet and may be updated. Github actions have been set to update the '.whl' file automatically. In order to retrieve the last updates, first pull the latest version of the repository and then reinstall the the '.whl' file. Make sure to be in your virtual environment.

```
git pull
```

**Linux and MacOS:**
```
source .venv/bin/activate
pip install dist/packaging_baboontrack-0.0.1-py3-none-any.whl --force-reinstall
```

**Windows:**
```
.\.venv\Scripts\activate
pip install .\dist\packaging_baboontrack-0.0.1-py3-none-any.whl --force-reinstall
```

## Documentation

Doc is built automatically from source code. It is deployed on `gh-pages` branch.
Link: https://ccp-eva.github.io/packaging_baboontrack/

Once this repository published, the documentation will need to be published too (Settings -> Pages).
Doc is also available in the `docs` folder of this repository.

## Citation

If you use this program, please cite this work:

TODO