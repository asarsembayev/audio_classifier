import os
import argparse
from tqdm import tqdm

import pandas as pd
import yaml

import torch
from torch.utils.data import Dataset, DataLoader
from dataset import AudioDataset
from torchaudio.transforms import MelSpectrogram, MFCC
from torchsummary import summary

from cnn import CNNNetwork


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


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
	for X, y in data_loader:
		X, y = X.to(device), y.to(device)

		# calculate loss
		prediction = model(X)
		loss = loss_fn(prediction, y)

		# backpropagate error and update weights
		optimiser.zero_grad()
		loss.backward()
		optimiser.step()

	return loss.item()


def main(src_dir, train_labels_path, test_labels_path, transforms):
	#define device {cpu | cuda}
	if torch.cuda.is_available():
		device = 'cuda'
	else:
		device = 'cpu'
	print(f'{device} is available')

	# read configs
	audio_configs = read_configs(os.path.join('configs', 'train.yaml'))
	configs = read_configs(os.path.join('configs', 'features.yaml'))
	configs = {**audio_configs, **configs}


	# define feature extraction
	transform = get_feature(transforms, configs=configs)

	TRAIN_DURATION = configs['train']['train_duration']
	BATCH_SIZE = configs['train']['batch_size']

	# datasets and dataloaders
	train_dataset = AudioDataset(src_dir, train_labels_path, transforms=transform, \
		target_sample_rate=configs['audio']['sr'], train_duration=TRAIN_DURATION, device=device)
	test_dataset = AudioDataset(src_dir, test_labels_path, transforms=transform, \
		target_sample_rate=configs['audio']['sr'], train_duration=TRAIN_DURATION, device=device)

	train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
	test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

	# train loop
	# input_shape = next(iter(train_dataloader))[0][0].shape
	# input_shape = input_shape[1] * input_shape[2]
	# num_classes = pd.read_csv(train_labels_path, sep='\t').label.nunique()
	# print(f'{num_classes} , {input_shape}')

	# define network
	cnn = CNNNetwork(2304, 10).to(device)

	# define optimiser and loss function
	loss_fn = torch.nn.CrossEntropyLoss()
	optimiser = torch.optim.Adam(cnn.parameters(), lr=configs['train']['learning_rate'])

	num_epochs = configs['train']['epochs']
	progress = tqdm(range(num_epochs))
	for i in progress:
		loss = train_single_epoch(cnn, train_dataloader, loss_fn, optimiser, device)
		progress.set_description(f'epoch: {i} | loss: {loss}')


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("src_dir", type=str, help="[input] path to directory with raw audio files")
	parser.add_argument("train_labels_path", type=str, help="[input] path to csv w/ labels")
	parser.add_argument("test_labels_path", type=str, help="[input] path to csv w/ labels")
	parser.add_argument("--transforms", type=str, help="[input] transforms", default=None)
	#TODO: add device param
	args = parser.parse_args()
	main(args.src_dir, args.train_labels_path, args.test_labels_path, args.transforms)
