import torch.utils.data as data_utils
import librosa
import numpy as np
class Data(data_utils.Dataset):
    def __init__(self, files):
        self.files = [i.strip() for i in open(files).readlines()]
    def __getitem__(self, idx):
        # sound, sample_rate = torchaudio.load(self.files[idx])
        sound, sample_rate = librosa.load(self.files[idx], mono=False)
        mono = librosa.to_mono(sound)
        len_frame = 1024
        len_hop = 1024 // 4
        spectrogram_mono = librosa.stft(mono, n_fft=len_frame, hop_length=len_hop)
        spectrogram_nonvocal = librosa.stft(sound[0], n_fft=len_frame, hop_length=len_hop)
        spectrogram_mono = spectrogram_mono.astype(np.float32)
        spectrogram_nonvocal = spectrogram_nonvocal.astype(np.float32)
        
        return spectrogram_mono, spectrogram_nonvocal
    def __len__(self):
        return len(self.files)

batch_size = 8
train_dataset = Data('data_train.txt')
# test_dataset = Data('train')
train_loader = data_utils.DataLoader(train_dataset, batch_size)
# test_loader = data_utils.DataLoader(test_dataset, batch_size)

if __name__ == '__main__':
    data = Data('data_train.txt')
    print(data.__len__())
    for i in train_loader:
        print(i)