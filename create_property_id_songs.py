from tqdm import tqdm
import os

from music21 import converter
from music21.note import Note
from music21.chord import Chord

from fractions import Fraction

import pickle

import torch

import warnings
warnings.filterwarnings("ignore")

def get_songs(folder_path: str) -> list:
    songs = []
    failed_songs = []
    paths = []
    for artist in tqdm(os.listdir(folder_path)):
        artist_path = folder_path + artist + "/"
        for song_name in os.listdir(artist_path):
            song_title = song_name.split(".")[0]
            try:
                song = converter.parse(artist_path + song_name)
                songs.append(song.flat.notes)
                paths.append(artist + "_" + song_title)
            except:
                failed_songs.append(song_title)

    return songs, failed_songs, paths


# load and parse songs
print("Starting to load and parse songs")
folder_path = "./midis/"
loaded_songs, failed_songs, paths = get_songs(folder_path)
print(f"Loaded ({len(loaded_songs)}) songs")
print(f"Failed to load ({len(failed_songs)}) song files: {failed_songs}")


# Functions for rounding and properties
def round_value(val: float, denom: int, max_val: Fraction = None) -> Fraction:
    if max_val and val > max_val:
        return max_val
    return Fraction(int(round(denom*val)), denom)

def note2properties(note: Note, vol_denom: int) -> tuple:
    return str(note.pitch), round_value(note.volume.velocityScalar, vol_denom)


def assign_property_ids(properties: list, dictionaries: list) -> None:
    
    num_properties = len(properties)
    curr_ids = torch.zeros(size=(1,num_properties), dtype=torch.int64)

    for i,prop in enumerate(properties):
        if prop not in dictionaries[i]:
            dictionaries[i][prop] = len(dictionaries[i])
        curr_ids[0,i] = dictionaries[i][prop]

    return curr_ids
        


# Create dictionaries and property id songs
print("Creating corpus and assigning ids to songs")
pitch2id = {"<SOS>": 0, "<EOS>": 1}
duration2id = {"<SOS>": 0, "<EOS>": 1}
offset2id = {"<SOS>": 0, "<EOS>": 1}
volume2id = {"<SOS>": 0, "<EOS>": 1}
dictionaries = [pitch2id, duration2id, offset2id, volume2id]

property_id_songs = []

num_properties = 4
max_time_denom = 12
max_volume_denom = 8
max_time = Fraction(4,1)

for song in tqdm(loaded_songs):

    property_id_song = torch.zeros(size=(1, num_properties), dtype=torch.int64)

    prev_offset = 0
    for item in song:

        # get relative offset info
        curr_offset = round_value(item.offset, max_time_denom)
        relative_offset = round_value(curr_offset - prev_offset, max_time)
        prev_offset = curr_offset

        # get item duration
        duration = round_value(item.quarterLength, max_time_denom, max_time)

        if isinstance(item, Note):
            pitch, volume = note2properties(item, max_volume_denom)
            properties = [pitch, duration, relative_offset, volume]
            curr_ids = assign_property_ids(properties, dictionaries)
            property_id_song = torch.concat([property_id_song, curr_ids], dim=0)
        elif isinstance(item, Chord):
            for i,note in enumerate(item):
                pitch, volume = note2properties(note, max_volume_denom)
                properties = [pitch, duration, relative_offset if i==0 else 0, volume]
                curr_ids = assign_property_ids(properties, dictionaries)
                property_id_song = torch.concat([property_id_song, curr_ids], dim=0)
    property_id_song = torch.concat([property_id_song, torch.ones(size=(1,num_properties), dtype=torch.int64)])
    property_id_songs.append(property_id_song)


# Dictionary of Properties to Unique Note
full_ids = {
    "0-0-0-0": 0,
    "1-1-1-1": 1,
}

for song in tqdm(property_id_songs):
    for item in song:
        item_str = "-".join([str(x) for x in item.tolist()])
        if not item_str in full_ids:
            full_ids[item_str] = len(full_ids)


# save songs and corpora dicts
print("Saving property id songs and corpus")

dicts_folder = "ids/"
if not os.path.exists(dicts_folder):
    os.makedirs(dicts_folder)

with open(dicts_folder + "pitch.pkl", "wb") as f:
    pickle.dump(pitch2id, f)
with open(dicts_folder + "duration.pkl", "wb") as f:
    pickle.dump(duration2id, f)
with open(dicts_folder + "offset.pkl", "wb") as f:
    pickle.dump(offset2id, f)
with open(dicts_folder + "volume.pkl", "wb") as f:
    pickle.dump(volume2id, f)
with open(dicts_folder + "full.pkl", "wb") as f:
    pickle.dump(full_ids, f)


property_id_songs_folder = "property_id_songs/"
if not os.path.exists(property_id_songs_folder):
    os.makedirs(property_id_songs_folder)
for i,property_id_song in tqdm(enumerate(property_id_songs)):
    torch.save(property_id_song, property_id_songs_folder + f"{paths[i]}.pt")


