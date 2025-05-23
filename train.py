import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from datetime import datetime
import model
import get_train_data
import pickle
import math
import time
import os

from config import *

os.makedirs("raw", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("dataset", exist_ok=True)

transform = transforms.Compose([
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
# ---------- DATA ----------

print("---------- loading data ----------")
get_train_data.load_data(IMAGES_URL, CAPTIONS_URL, IMAGES_ZIP_PATH, CAPTIONS_ZIP_PATH)
vocab = get_train_data.Vocabulary(min_freq=1)
caption_dict, max_caption_length, all_captions = get_train_data.get_caption_dict(get_train_data.read_captions(CAPTIONS_ZIP_PATH))

print("building vocab...")
vocab.build_vocabulary(all_captions)
vocab_size = len(vocab.stoi)
dataset = get_train_data.Flickr8k(IMAGES_ZIP_PATH, CAPTIONS_ZIP_PATH, transform, vocab)
print(f"dataloader workers: {os.cpu_count() // 2}")
dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2, drop_last=True)
with open(VOCAB_PATH, "wb") as f:
    pickle.dump(vocab, f)

print("done!")

# ---------- CHECKPOINTS ----------

def save_checkpoint(model, optimizer, step, path=CHECKPOINT_PATH):
    torch.save({
        'model':      model.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'step':       step,
    }, path)

def load_checkpoint(model, optimizer=None, path=CHECKPOINT_PATH):
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt['model'])
    if optimizer:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['step']

# ---------- HELPERS ----------

def get_lr(it):
    if it < WARMUP_STEPS:
        return MAX_LR * (it + 1) / WARMUP_STEPS
    
    if it > MAX_STEPS:
        return MIN_LR

    decay_ratio = (it - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 *( 1.0 +  math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)

def closest_power(n):
    if n < 1:
        raise ValueError("input must be a positive integer.")
    return 1 << (n - 1).bit_length()

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def print_config(model):
    encoder_total, encoder_trainable = count_parameters(model.encoder)
    print(f"encoder, parameters: {encoder_total:,}, trainable: {encoder_trainable:,}")
    decoder_total, decoder_trainable = count_parameters(model.decoder)
    print(f"decoder, parameters: {decoder_total:,}, trainable: {decoder_trainable:,}")
    full_total, full_trainable = count_parameters(model)
    print(f"total, parameters: {full_total:,}, trainable: {full_trainable:,}")

def loader(data_loader):
    while True:
        for batch in data_loader:
            yield batch

# ---------- SETUP ----------

print("\n---------- configuring model ----------")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"current device: {device.upper()}")
modelconfig = model.ITTConfig()
modelconfig.vocab_size = closest_power(vocab_size)
ITT = model.ITT(modelconfig).to(device)
ITT = torch.compile(ITT)
torch.set_float32_matmul_precision("high")
print_config(ITT)

optimizer = ITT.configure_optimizers(weight_decay=0.1, learning_rate=MIN_LR, device=device)
data_iter = loader(dataloader)

print("done!")

# ---------- TRAINING ----------

print("\n---------- starting training ----------")
if os.path.exists(CHECKPOINT_PATH):
    start_step = load_checkpoint(ITT, optimizer, CHECKPOINT_PATH)
    print(f"loaded checkpoint at step {start_step}\n")
else:
    start_step = 0
try:
    for step in range(start_step, MAX_STEPS):
        ITT.train()
        image, caption = next(data_iter)
        image, caption = image.to(device), caption.to(device)
        t0 = time.time()

        optimizer.zero_grad()
        loss, _ = ITT.forward(image, caption)
        loss.backward()
        optimizer.step()

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0)*1000
        tokens_per_sec = (modelconfig.batch_size) / (t1 - t0)
        now = datetime.now()
        print(f"[{now.strftime("%Y-%m-%d %H:%M:%S")}] step: {step}/{MAX_STEPS}\tloss: {loss.item():.3f}\tdt: {dt:.3f} ms\timgs/sec: {tokens_per_sec:.3f}")

        if step % SAVE_EVERY == 0:
            save_checkpoint(ITT, optimizer, step, CHECKPOINT_PATH)
except KeyboardInterrupt:
    print("\nexiting safely...")
    print("saving model and clearing cache...")
    save_checkpoint(ITT, optimizer, step, CHECKPOINT_PATH)
    torch.cuda.empty_cache()
    print("done!")

if step == MAX_STEPS-1:
    save_checkpoint(ITT, optimizer, step, CHECKPOINT_PATH)
    
    