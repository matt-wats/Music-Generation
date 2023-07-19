import torch
import os
import numpy as np
import pickle
import music21

torch.manual_seed(42)
#----------------------------------------------------------------------------------------------------------------------------------------------------


max_song_length = 500
memory_length = 250


storage_device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}")


#----------------------------------------------------------------------------------------------------------------------------------------------------

# load id corpus
ids_folder = "./ids/"
with open(ids_folder + 'pitch.pkl', 'rb') as f:
    pitch = pickle.load(f)
with open(ids_folder + 'duration.pkl', 'rb') as f:
    duration = pickle.load(f)
with open(ids_folder + 'offset.pkl', 'rb') as f:
    offset = pickle.load(f)
with open(ids_folder + 'volume.pkl', 'rb') as f:
    volume = pickle.load(f)
with open(ids_folder + "full.pkl", "rb") as f:
    full2id = pickle.load(f)
full_list = list(full2id)
dictionaries = [pitch, duration, offset, volume]
num_properties = len(dictionaries)

# load trained model
training_folder = "./training_stats/"
model = torch.load(training_folder + "model.pt")
state = torch.load(training_folder + "model_state_29.pt")
model.load_state_dict(state)
model.eval()

#----------------------------------------------------------------------------------------------------------------------------------------------------

print("Creating song...")

# create song
created_song = torch.zeros(size=(1, num_properties), dtype=torch.int64, device=device)

end_token = False
while not end_token and created_song.size(0) < max_song_length:
    preds = model(created_song[-memory_length:])
    new_note_properties = [int(x) for x in full_list[preds[-1].argmax()].split("-")]
    new_note = torch.tensor(new_note_properties, dtype=torch.int64, device=device).view(1,num_properties)

    created_song = torch.concat([created_song, new_note], dim=0)

    if 0 in new_note or 1 in new_note:
        end_token = True

print(f"Song created with {created_song.size(0)} tokens ({max_song_length} max)")

#----------------------------------------------------------------------------------------------------------------------------------------------------

# turn property id song into midi

def PropertyIDs2Properties(prop_ids: torch.Tensor, corpora: list):

    lists = [list(corpus.keys()) for corpus in corpora]

    song_values = []

    for ids in prop_ids[1:-1]:
        pitch = lists[0][ids[0]]
        duration = lists[1][ids[1]]
        offset = lists[2][ids[2]]
        volume = lists[3][ids[3]]

        song_values.append([pitch, duration, offset, volume])

    return song_values


def Properties2Midi(props_list: list):
    stream = music21.stream.Stream()

    curr_offset = 0
    for props in props_list:
        chord = music21.chord.Chord(notes=props[0], quarterLength=props[1])
        chord.volume.velocityScalar = props[3]
        stream.insert(curr_offset, chord)
        curr_offset += props[2]

    return stream

#----------------------------------------------------------------------------------------------------------------------------------------------------

pred_properties = PropertyIDs2Properties(created_song, dictionaries)
pred_stream = Properties2Midi(pred_properties)

pred_stream.write('midi', fp="created_song.midi")