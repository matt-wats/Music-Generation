import torch
import os
from Model import TransformerModel
import numpy as np
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
import math
import random

torch.manual_seed(42)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# set manual variables

property_id_songs_folder = "./property_id_songs/"
ids_folder = "./ids/"


property_dims = [255, 64, 64, 1]
full_dim = np.sum(property_dims)


select_by_artist = False
song_artist = "chopin"
max_seq_len = 250
nearest_slice = 250


nhead = 6
num_layers = 6
dropout = 0.1


max_lr = 1e-3
min_lr = 1e-4
warmup_steps = 25
decay_steps = 225
num_epochs = warmup_steps + decay_steps


verbose = 10
batch_size = 25

train_percent = 0.5

storage_device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}")

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Load songs and corpus dictionaries
property_id_songs = []
for file_name in os.listdir(property_id_songs_folder):
    if not select_by_artist:
        property_id_song = torch.load(property_id_songs_folder + file_name, map_location=storage_device)
        property_id_songs.append(property_id_song)
    elif song_artist in file_name:
        property_id_song = torch.load(property_id_songs_folder + file_name, map_location=storage_device)
        property_id_songs.append(property_id_song)

random.seed(0)
random.shuffle(property_id_songs)
num_songs = len(property_id_songs)
cutoff = int(train_percent * num_songs)

train_id_songs = property_id_songs[:cutoff]
val_id_songs = property_id_songs[cutoff:]

with open(ids_folder + "pitch.pkl", "rb") as f:
    pitch2id = pickle.load(f)
with open(ids_folder + "duration.pkl", "rb") as f:
    duration2id = pickle.load(f)
with open(ids_folder + "offset.pkl", "rb") as f:
    offset2id = pickle.load(f)
with open(ids_folder + "volume.pkl", "rb") as f:
    volume2id = pickle.load(f)
with open(ids_folder + "full.pkl", "rb") as f:
    full2id = pickle.load(f)
num_unique_notes = len(full2id)
property_lengths = [len(pitch2id), len(duration2id), len(offset2id), len(volume2id)]
num_properties = len(property_lengths)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Select training songs and make them into batched tensors
train_slices = []
for song_idx, song in enumerate(train_id_songs):
    start_spots = np.arange(0, song.size(0) - max_seq_len - 1, nearest_slice)
    for start_spot in start_spots :
        train_slices.append((song_idx, start_spot))
num_train_slices = len(train_slices)
print(f"Number of training slices: {num_train_slices}")

# Select validation songs and make them into batched tensors
val_slices = []
for song_idx, song in enumerate(val_id_songs):
    start_spots = np.arange(0, song.size(0) - max_seq_len - 1, nearest_slice)
    for start_spot in start_spots :
        val_slices.append((song_idx, start_spot))
num_val_slices = len(val_slices)
print(f"Number of validation slices: {num_val_slices}")


num_batches = num_train_slices // batch_size

def get_batch(slices_indices: list, songs: list, slices: list) -> torch.Tensor:
    input_batch = torch.ones(size=(batch_size, max_seq_len, num_properties), dtype=torch.int32, device=device)
    target_properties = torch.ones(size=(batch_size, max_seq_len, num_properties), dtype=torch.int64, device=device)
    target_batch = torch.ones(size=(batch_size, max_seq_len), dtype=torch.int64, device=device)

    for i in range(batch_size):
        song_idx, start_spot = slices[slices_indices[i]]
        song = songs[song_idx]
        song_len = song.size(0)
        input_batch[i, :min(song_len-1, max_seq_len)] = song[start_spot:min(song_len-1,start_spot+max_seq_len)]
        target_properties[i, :min(song_len-1, max_seq_len)] = song[start_spot+1:start_spot+max_seq_len+1]
        target_batch[i] = torch.tensor([full2id["-".join([str(x) for x in tgt_props.tolist()])] for tgt_props in target_properties[i]])



    return input_batch, target_batch


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Set model parameters
model = TransformerModel(property_dims, property_lengths, num_unique_notes, nhead=nhead, num_layers=num_layers, dim_feedforward=4*full_dim, dropout=dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0)
criterion = torch.nn.CrossEntropyLoss()

def get_lr(epoch: int) -> float:
    if epoch < warmup_steps:
        return max_lr * (epoch+1) / (warmup_steps+1)
    elif epoch > warmup_steps + decay_steps:
        return min_lr

    decay_ratio = (epoch - warmup_steps) / (decay_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

losses = []
val_losses = []
lrs = []

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# train model
training_folder = "./training_stats/"
if not os.path.exists(training_folder):
    os.makedirs(training_folder)
print("Beginning model training")
for epoch in (range(num_epochs)):

    lr = get_lr(epoch)
    for param in optimizer.param_groups:
        param['lr'] = lr
    lrs.append(lr)

    r = torch.randperm(num_train_slices)
    for batch_num in (range(num_batches)):
        batch_inputs, batch_targets = get_batch(r[batch_num*batch_size:(batch_num+1)*batch_size], train_id_songs, train_slices)

        optimizer.zero_grad()
        pred = model(batch_inputs)

        loss = criterion(pred.transpose(1,2), batch_targets)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    if (epoch+1) % verbose == 0 or epoch == 0:

        batch_inputs, batch_targets = get_batch(torch.arange(num_val_slices), val_id_songs, val_slices)
        with torch.no_grad():
            pred = model(batch_inputs)
            loss = criterion(pred.transpose(1,2), batch_targets)
            val_losses.append(loss.item())
        torch.save(model.state_dict(), training_folder + f"model_state_{epoch}.pt")

        print(f"Epoch #{epoch+1}/{num_epochs} losses: {np.mean(losses[-num_batches:])}, val losses: {val_losses[-1]}")

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

torch.save(model, training_folder + "model.pt")
torch.save(model.state_dict(), training_folder + "model_state.pt")

torch.save(optimizer, training_folder + "optimizer.pt")
torch.save(optimizer.state_dict(), training_folder + "optimizer_state.pt")

with open(training_folder + "losses.pkl", "wb") as f:
    pickle.dump(losses, f)
with open(training_folder + "val_losses.pkl", "wb") as f:
    pickle.dump(val_losses, f)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# plot training
fig, axs = plt.subplots(3, figsize=(10,10))
axs[0].semilogy([np.mean(losses[j:j+num_batches]) for j in range(0,num_epochs*num_batches,num_batches)])
axs[1].semilogy(val_losses)
axs[2].plot(lrs)
plt.show()