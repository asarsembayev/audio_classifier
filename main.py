import os
import argparse
from tqdm import tqdm

import yaml

import torch
from torch.utils.data import Dataset, DataLoader
from dataset import AudioDataset
from torchaudio.transforms import MelSpectrogram, MFCC


def read_configs(path):
	with open(path) as handle:
		configs = yaml.load(handle, Loader=yaml.FullLoader)
	return configs


def get_feature(feature_type, configs=None):
	if configs:
		features = {
			'mel': MelSpectrogram(
				sample_rate = configs['audio']['sr'], n_fft = configs['mel']['n_fft'], hop_length = configs['mel']['hop_length']
				),
			'mfcc': ''
		}
	else:
		features = {
			'mel': MelSpectrogram(),
			'mfcc': MFCC()
		}

	return features[feature_type]


def main(src_dir, labels_path, transforms):
	audio_configs = read_configs(os.path.join('configs', 'train.yaml'))
	configs = read_configs(os.path.join('configs', 'features.yaml'))
	configs = {**audio_configs, **configs}
	transform = get_feature(transforms, configs=configs)

	TRAIN_DURATION = configs['train']['train_duration']

	train_dataset = AudioDataset(src_dir, labels_path, transforms=transform, \
		target_sample_rate=configs['audio']['sr'], train_duration=TRAIN_DURATION)
	# test_dataset = AudioDataset(src_dir, labels_path, transforms=None)

	train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=False)
	# test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

	for sample in train_dataloader:
		print(sample)
		exit(0)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("src_dir", type=str, help="[input] path to directory with raw audio files")
	parser.add_argument("labels_path", type=str, help="[input] path to csv w/ labels")
	parser.add_argument("--transforms", type=str, help="[input] transforms", default=None)
	args = parser.parse_args()
	main(args.src_dir, args.labels_path, args.transforms)
