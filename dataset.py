import torch.utils.data as data_utils
import librosa

class Data(data_utils.Dataset):
    def __init__(self, files):
        self.files = [i.strip() for i in open(files).readlines()]
    def __getitem__(self, idx):
        # sound, sample_rate = torchaudio.load(self.files[idx])
        sound, sample_rate = librosa.load('a.wav', mono=False)
        mono = librosa.to_mono(sound)
        len_frame = 1024
        len_hop = 1024 // 4
        spectrogram_mono = librosa.stft(mono, n_fft=len_frame, hop_length=len_hop)
        spectrogram_accompaniment = librosa.stft(sound[0], n_fft=len_frame, hop_length=len_hop)
        return spectrogram_mono, spectrogram_accompaniment
    def __len__(self):
        return len(self.files)

if __name__ == '__main__':
    data = Data('data_train.txt')
    print(data.__len__())
    for i in range(data.__len__()):
        print(data.__getitem__(i))