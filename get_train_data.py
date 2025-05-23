import os
import requests
import zipfile
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
import io

import re
import string
from collections import Counter


def download_file(url, path, block_size, description):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    with tqdm(total=total_size, unit="B", unit_scale=True, desc=description) as progress_bar:
        with open(path, 'wb') as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)

def load_data(image_url, captions_url, image_zip_path, captions_zip_path):
    block_size = 1024

    if not os.path.exists(image_zip_path):
        download_file(image_url, image_zip_path, block_size, "Downloading Flickr8k_Dataset.zip")
    else:
        print("loading existing Flickr8k_Dataset.zip...")

    if not os.path.exists(captions_zip_path):
        download_file(captions_url, captions_zip_path, block_size, "Downloading Flickr8k_text.zip")
    else:
        print("loading existing Flickr8k_text.zip...")


def read_captions(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open("Flickr8k.token.txt") as f:
            return f.read().decode('utf-8')

def get_caption_dict(captions):
    caption_dict = {}
    max_caption_length = 0
    all_captions = []

    for line in captions.split("\n"):
        if line:
            file_name, caption = line.split("\t")
            file_name = file_name[:-2]
            if file_name not in caption_dict:
                caption_dict[file_name] = [caption]
            else:
                caption_dict[file_name].append(caption)

            max_caption_length = max(len(caption.split(" ")), max_caption_length)
            all_captions.append(caption)

    return caption_dict, max_caption_length, all_captions


class Flickr8k(Dataset):
    def __init__(self, images_zip_path, captions_zip_path, transform, vocab):
        self.images_zip_path = images_zip_path
        self.captions_zip_path = captions_zip_path
        self.transform = transform
        self.vocab = vocab
        self.caption_dict, self.max_caption_length, self.all_captions = get_caption_dict(read_captions(captions_zip_path))

        with zipfile.ZipFile(images_zip_path, 'r') as archive:
            self.image_files = [file_name for file_name in archive.namelist() if file_name.lower().endswith(('.jpg', '.jpeg', '.png'))]

        self.data_pairs = []
        for file_name in self.image_files:
            clean_name = file_name[18:]
            if clean_name in self.caption_dict:
                for caption in self.caption_dict[clean_name]:
                    self.data_pairs.append((file_name, caption))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        file_name, caption = self.data_pairs[idx]

        with zipfile.ZipFile(self.images_zip_path, 'r') as archive:
            with archive.open(file_name) as file:
                img = Image.open(io.BytesIO(file.read())).convert("RGB")
                img = self.transform(img)
        
        caption_indices = self.vocab.encode_caption(caption, self.max_caption_length)
        caption_tensor = torch.tensor(caption_indices, dtype=torch.long)

        if len(caption_indices) == 0: 
            raise ValueError("Empty caption found")

        return img, caption_tensor

class Vocabulary:
    def __init__(self, min_freq=4):
        self.min_freq = min_freq
        self.word_counts = Counter()
        self.itos = {}  # Index to string mapping
        self.stoi = {}  # String to index mapping
    
    def tokenize(self, text):
        text = text.lower()
        text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
        return text.split()
    
    def build_vocabulary(self, captions):
        for caption in captions:
            tokens = self.tokenize(caption)
            self.word_counts.update(tokens)
        
        self.vocabulary = {word for word, count in self.word_counts.items() if count >= self.min_freq}
        
        sorted_vocab = ["<pad>", "<unk>", "<start>", "<end>"] + sorted(self.vocabulary)
        self.itos = {idx: word for idx, word in enumerate(sorted_vocab)}
        self.stoi = {word: idx for idx, word in self.itos.items()}
        
        return self.vocabulary
    
    def encode_caption(self, caption, max_length=50):
        tokens = ["<start>"] + self.tokenize(caption) + ["<end>"]
        token_indices = [self.stoi.get(token, self.stoi["<unk>"]) for token in tokens]
        
        # Pad or truncate to max_length
        if len(token_indices) < max_length:
            token_indices += [self.stoi["<pad>"]] * (max_length - len(token_indices))
        else:
            token_indices = token_indices[:max_length]
        
        return token_indices
    
    
