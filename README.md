# decode-images
Repository for code review. The `-epo.fif` datafiles on shared drive, and should be placed in the `derivatives/epochs/loc` and `derivatives/epochs/main` folders manually.

## Folder structure
```
decode-images
  |---analysis
  |      |---JY_glhmm_utils.py
  |      |---glhmm_decode_matintask.ipynb
  |      |--- ...
  |
  |---derivatives
  |      |---epochs
  |      |      |---loc
  |      |      |    |---sub401_stimOn-epo.fif
  |      |      |    |---subXXX_stimOn-epo.fif
  |      |      |
  |      |      |---main
  |      |      |    |---sub401_stim1On-epo.fif
  |      |      |    |---subXXX_stim1On-epo.fif  
```
## Dependencies
- python==3.8.19
- mne==1.6.1
- glhmm==0.2.6
- numpy==1.24.4
## Usage
- `JY_glhmm_utils.py` contains utility functions I wrote to prepare data for training, and to get decoders' performance metrics.
- `glhmm_decode_maintask.ipynb` describes the analysis and shows the step-by-step outputs. 
