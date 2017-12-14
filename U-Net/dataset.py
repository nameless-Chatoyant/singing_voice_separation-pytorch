import torch.utils.data as data_utils

import torchaudio

import librosa

wav, sr = torchaudio.load('a.wav')
print(wav.shape)
sound, sample_rate = torchaudio.load('a.wav', mono=False)

print(wav.shape)

class Data(data_utils.Dataset):
    def __init__(self, root, transform=None, train=True):
        pass
    def __getitem__(self, idx):
        sound, sample_rate = torchaudio.load('a.wav', mono=False)
        pass
    def __len__(self):
        pass

if __name__ == '__main__':
    data = Data()
    for i in range(data.__len__):
        print(data.__getitem__(i))