# audio_classifier

## Description

This is an Audio Classifier suitable for variable audio classification tasks (i.g. Speech Emotion Classification)

## Table of Contents (Optional)

- [Installation](#installation)
- [Usage](#usage)
- [Credits](#credits)
- [License](#license)

## Installation

Please install necessary libraries with
```
pip install -r requirements.txt
```

## Usage
```
usage: main.py [-h] [--transforms TRANSFORMS]
               [--configs_path CONFIGS_PATH]
               src_dir labels_path

positional arguments:
  src_dir               [input] path to directory with raw audio files
  labels_path           [input] path to csv w/ labels

optional arguments:
  -h, --help            show this help message and exit
  --transforms TRANSFORMS
                        [input] transforms
  --configs_path CONFIGS_PATH
                        [input] configs_path
```