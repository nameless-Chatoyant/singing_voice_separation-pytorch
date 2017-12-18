# Overview

# Requirements
- pytorch
- librosa
    ```shell
    pip3 install librosa
    ```
- librosa
# Usage
- data helper
    As the model only takes wave files as inputs, it will collect all supported audio file paths in preprocessing procedure.
    ```bash
    python3 preprocess.py --dir <dir-path>
    ```
- training
    ```bash
    python3 main.py --model <model-name> --train --train_manifest <txt-path>
    ```
- evaluation
    ```bash
    python3 main.py --model <model-name> --eval --eval_manifest <txt-path>
    ```

- predict
    ```shell
    python3 main.py --model <model-name> --predict <model-name>
    ```
# Results
our result(paper/official repo result) on iKala
| Model | NSDR(Vocal) | NSDR(Instrumental) | SIR(Vocal) | SIR(Instrumental) | SAR(Vocal) | SAR(Instrumental) |
|:-----:|:-----------:|:------------------:|:----------:|:-----------------:|:----------:|:-----------------:|
| U-Net | (11.094) | (14.435) | (23.960) | (21.832) | (17.715) | (14.120) |
| SVSGAN | (-) | (-) | (23.70) | (-) | (14.10) | (-) |
| GRU-RIS | (-) | (-) | (23.70) | (-) | (14.10) | (-) |

# Reference
| Model | Original Paper | Official Repo |
|:-----:|:-----:|:-----:|
| U-Net | [Singing Voice Separation with Deep U-Net Convolutional Networks](https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf)| - |
| SVSGAN | [SVSGAN: Singing Voice Separation via Generative Adversarial Network](https://arxiv.org/abs/1710.11428)| - |
| MSS | [Singing Voice Separation via Recurrent Inference and Skip-Filtering Connections](https://arxiv.org/abs/1711.01437) | [Js-Mim/mss_pytorch](https://github.com/Js-Mim/mss_pytorch) |

# Troubleshooting
1. NoBackendError audioread
    ```shell
    ```