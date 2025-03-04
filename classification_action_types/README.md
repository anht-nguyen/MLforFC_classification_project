
## Models and Techniques

The project includes the following models and techniques:

- **COH**: Coherence
- **iCOH**: Imaginary Coherence
- **PDC**: Partial Directed Coherence
- **PLV**: Phase Locking Value
- **Spectral Granger**: Spectral Granger Causality

Classes in the dataset: 0 - air punching, 1 - forward-backward swing, 2 - medial-lateral swing, 3 - overhead arm raise, 4 - waving


## Installation

To install the required packages, run:

```sh
pip install -r requirements.txt
``` 

To log the necessary packages, install `pipreqs`:
```
pip install pipreqs
```
Generate a Minimal `requirements.txt`:
``` 
pipreqs . --force
```


## Running classification algorithms:

Before running the classification, check the `config.py` script which store most important information, especially training parameters.

ML models: run `python ML_models_training_validating.py` 

Tip: Use `nohup python <script.py> &` to run python script in background (non tmux option)

For each of the DL models, run the corresponding shell script with tmux:

```
sh ./autorun_<model>_model_singleFC.sh <tmux session name>
```

To open the tmux session: `tmux -a -t <session name>`