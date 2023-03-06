import os

import pandas as pd

import torch
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram, Resample
from torchaudio import load as load_audio


class AudioDataset(Dataset):
	"""a custom pytorch Dataset for audio classifier"""
	def __init__(self, src_dir, labels_path, transforms, target_sample_rate, train_duration):
		self.src_dir = src_dir
		self.labels = pd.read_csv(labels_path, sep='\t')
		self.transforms = transforms
		self.target_sample_rate = target_sample_rate
		self.num_samples = int(train_duration * target_sample_rate)


	def __len__(self):
		return len(self.labels)


	def __getitem__(self, idx):
		audio_path = os.path.join(self.src_dir, self.labels.iloc[idx, 0])
		label = self.labels.iloc[idx, 1]

		signal, sr = load_audio(audio_path, normalize=True)
		signal = self._resample_if_necessary(signal, sr)
		signal = self._mix_down_if_necessary(signal)
		signal = self._cut_if_necessary(signal)
		signal = self._right_pad_if_necessary(signal)
		signal = self.transforms(signal)

		return signal, label


	def _cut_if_necessary(self, signal):
		if signal.shape[1] > self.num_samples:
			signal = signal[:, :self.num_samples]
		return signal


	def _right_pad_if_necessary(self, signal):
		length_signal = signal.shape[1]
		if length_signal < self.num_samples:
			num_missing_samples = self.num_samples - length_signal
			last_dim_padding = (0, num_missing_samples)
			signal = torch.nn.functional.pad(signal, last_dim_padding)
		return signal


	def _resample_if_necessary(self, signal, sr):
		if sr != self.target_sample_rate:
			resampler = Resample(sr, self.target_sample_rate)
			signal = resampler(signal)
		return signal


	def _mix_down_if_necessary(self, signal):
		if signal.shape[0] > 1:
			signal = torch.mean(signal, dim=0, keepdim=True)
		return signal
