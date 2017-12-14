
[Singing Voice Separation with Deep U-Net Convolutional Networks](https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf)

# Requirements

# Usage
1. collect data
As the model only takes wave files as inputs, it will collect all supported audio file paths in preprocessing procedure.
```bash
python3 preprocess.py --dir /path/to/data/
```
2. train model
```bash
python3 train.py
```
3. predict
# Results

#
NoBackendError audioread