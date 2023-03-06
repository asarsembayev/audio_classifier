import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from dataset import AudioDataset


def main(src_dir, labels_path, transforms):
	train_dataset = AudioDataset(src_dir, labels_path, transforms=None)
	test_dataset = AudioDataset(src_dir, labels_path, transforms=None)

	train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=False)
	test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

	for sample in train_dataloader:
		print(sample)
		exit(0)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("src_dir", type=str, help="[input] path to directory with raw audio files")
	parser.add_argument("labels_path", type=str, help="[input] path to csv w/ labels")
	parser.add_argument("--transforms", type=str, help="[input] transforms")
	args = parser.parse_args()
	main(args.src_dir, args.labels_path, args.transforms)