import requests

import bs4
from bs4 import BeautifulSoup

import os



# Functions for getting the artists and songs off the website

def get_artists(artists_url: str) -> list:

    response = requests.get(artists_url)
    content = response.content
    soup = BeautifulSoup(content, "html.parser")

    links = soup.find_all("a")

    artist_links = [link.get("href") for link in links if artist_link_criteria(link)]

    return artist_links

def get_songs(songs_url: str) -> list:

    # Send a request to the webpage and get its content
    response = requests.get(songs_url)
    content = response.content

    # Use BeautifulSoup to parse the content and find all links
    soup = BeautifulSoup(content, "html.parser")
    links = soup.find_all("a")

    # Filter the links to only include those with the specified file extension
    song_links = [link.get("href") for link in links if song_link_criteria(link)]

    return song_links


# Functions for criteria for if we should download

def song_link_criteria(link: bs4.element.ResultSet) -> bool:
    href = link.get("href")

    if href is None: return False
    if not href.endswith(".mid"): return False
    if not "format0" in href: return False
    return True

def artist_link_criteria(link: bs4.element.ResultSet) -> bool:
    title = link.get("title")
    
    if title is None: return False
    if not "Audio Files" in title: return False

    return True



# download
main_url = "http://www.piano-midi.de/"
artists_url = "http://www.piano-midi.de/midi_files.htm"

download_folder = "midis"
if not os.path.exists(download_folder):
    os.makedirs(download_folder)

print("Collecting artists...")
artists = get_artists(artists_url)

print("Beginning download of songs...")
for idx, artist in enumerate(artists):
    songs = get_songs(main_url + artist)
    folder_path = download_folder + "/" + artist.split(".")[0]
    for song_page in songs:

        song = song_page.split("/")[-1]

        # check if folder path exists, if not, create it
        # path is midis/<ARTIST>/
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # get the song from the website
        response = requests.get(main_url + song_page)
        file_content = response.content

        # save the file
        with open(folder_path + "/" + song, "wb") as f:
            f.write(file_content)

    print(f"Finished downloading: {artist.split('.')[0].capitalize()} ({idx+1}/{len(artists)})")
