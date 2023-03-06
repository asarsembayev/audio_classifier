import os

import pandas as pd

from torch.utils.data import Dataset


class AudioDataset(Dataset):
	"""a custom pytorch Dataset for audio classifier"""
	def __init__(self, src_dir, labels_path, transforms=None):
		self.src_dir = src_dir
		self.labels = pd.read_csv(labels_path, sep='\t')
		self.transforms = transforms


	def __len__(self):
		return len(self.labels)


	def __getitem__(self, idx):
		audio_path = os.path.join(self.src_dir, self.labels.iloc[idx, 0])
		label = self.labels.iloc[idx, 1]

		return {'audio': audio_path, 'label': label}
