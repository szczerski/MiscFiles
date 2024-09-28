import pandas as pd
import numpy as np
import librosa
import os
import logging
from functools import lru_cache
import seaborn as sns
import matplotlib.pyplot as plt


class SongSimilarity:
    # Define weights as a class attribute
    WEIGHTS = {
        "tempo": 1/5,
        "chroma": 1/5,
        "contrast": 1/5,
        "centroid": 1/5,
        "rolloff": 1/5
    }

    def __init__(self, songs_folder, sample_rate=22050, verbose=True):
        self.songs_folder = songs_folder
        self.sample_rate = sample_rate
        self.verbose = verbose
        self.songs = {}
        self.features = {}

        log_format = "%(message)s"
        logging.basicConfig(
            level=logging.INFO if verbose else logging.CRITICAL, format=log_format
        )

        self.load_songs()
        self.extract_features()

    def load_songs(self):
        for file in os.listdir(self.songs_folder):
            if file.endswith(".mp3"):
                path = os.path.join(self.songs_folder, file)
                self.songs[file] = librosa.load(path, sr=self.sample_rate)[0]
                logging.info(f"Loaded {file}")

    @lru_cache(maxsize=None)
    def extract_features(self):
        for name, y in self.songs.items():

            tempo, _ = librosa.beat.beat_track(y=y, sr=self.sample_rate)

            chroma = librosa.feature.chroma_stft(y=y, sr=self.sample_rate)

            contrast = librosa.feature.spectral_contrast(y=y, sr=self.sample_rate)

            centroid = librosa.feature.spectral_centroid(y=y, sr=self.sample_rate)

            rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sample_rate)

            self.features[name] = {
                "tempo": float(
                    np.squeeze(tempo)
                ),
                "chroma": np.mean(chroma, axis=1),
                "contrast": np.mean(contrast, axis=1),
                "centroid": float(np.mean(centroid)),
                "rolloff": float(np.mean(rolloff))
            }
            logging.info(f"Extracted features for {name}")

    def compare_songs(self, song1, song2):
        features1 = self.features[song1]
        features2 = self.features[song2]

        similarities = {}

        tempo_diff = abs(features1["tempo"] - features2["tempo"])
        similarities["tempo"] = 1 / (1 + tempo_diff / 10)

        chroma_dist = np.linalg.norm(features1["chroma"] - features2["chroma"])
        similarities["chroma"] = 1 / (1 + chroma_dist / 5)

        contrast_dist = np.linalg.norm(features1["contrast"] - features2["contrast"])
        similarities["contrast"] = 1 / (1 + contrast_dist / 5)

        centroid_diff = abs(features1["centroid"] - features2["centroid"])
        similarities["centroid"] = 1 / (1 + centroid_diff / 1000)

        rolloff_diff = abs(features1["rolloff"] - features2["rolloff"])
        similarities["rolloff"] = 1 / (1 + rolloff_diff / 1000)

        total_similarity = sum(
            self.WEIGHTS[feature] * similarity
            for feature, similarity in similarities.items()
        )

        return float(total_similarity)

def create_combined_plot(similarity, most_similar, similarity_matrix, songs):
    fig = plt.figure(figsize=(18, 8))
    
    # Spider chart
    ax1 = fig.add_subplot(121, polar=True)
    features = list(SongSimilarity.WEIGHTS.keys())
    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False)
    
    similarities = {}
    for feature in features:
        if feature == 'tempo':
            tempo_diff = abs(similarity.features[most_similar[0]][feature] - similarity.features[most_similar[1]][feature])
            similarities[feature] = 1 / (1 + tempo_diff / 10)
        elif feature in ['centroid', 'rolloff']:
            diff = abs(similarity.features[most_similar[0]][feature] - similarity.features[most_similar[1]][feature])
            similarities[feature] = 1 / (1 + diff / 1000)
        else:
            dist = np.linalg.norm(similarity.features[most_similar[0]][feature] - similarity.features[most_similar[1]][feature])
            similarities[feature] = 1 / (1 + dist / 5)
    
    values = [similarities[feature] for feature in features]
    values.append(values[0])
    angles = np.concatenate((angles, [angles[0]]))
    
    ax1.plot(angles, values)
    ax1.fill(angles, values, alpha=0.25)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(features)
    ax1.set_ylim(0, 1)
    ax1.set_title(f"Feature Similarities: {most_similar[0]} vs {most_similar[1]}")
    
    # Heatmap
    ax2 = fig.add_subplot(122)
    sns.heatmap(similarity_matrix, annot=True, cmap="YlGnBu", xticklabels=songs, yticklabels=songs, ax=ax2)
    ax2.set_title("Song Similarity Heatmap")
    
    plt.tight_layout()
    plt.savefig("song_similarity_analysis.png")
    plt.close()


def main():
    songs_folder = 'songs'
    similarity = SongSimilarity(songs_folder)
    
    songs = list(similarity.songs.keys())
    
    similarity_matrix = np.zeros((len(songs), len(songs)))
    
    most_similar = (None, None, 0)
    least_similar = (None, None, 1)
    
    for i in range(len(songs)):
        for j in range(len(songs)):
            song1 = songs[i]
            song2 = songs[j]
            
            sim_score = similarity.compare_songs(song1, song2)
            similarity_matrix[i, j] = sim_score
            
            if i != j:
                if sim_score > most_similar[2]:
                    most_similar = (song1, song2, sim_score)
                if sim_score < least_similar[2]:
                    least_similar = (song1, song2, sim_score)
    
    create_combined_plot(similarity, most_similar, similarity_matrix, songs)
    
    print("Combined song similarity analysis saved as 'song_similarity_analysis.png'")
    
    print(f"\nMost similar pair: {most_similar[0]} and {most_similar[1]}")
    print(f"Similarity score: {most_similar[2]:.4f}")
    
    print(f"\nLeast similar pair: {least_similar[0]} and {least_similar[1]}")
    print(f"Similarity score: {least_similar[2]:.4f}")
    
    if most_similar[2] > 0.8:
        print(f"\nDetailed comparison between {most_similar[0]} and {most_similar[1]}:")
        features1 = similarity.features[most_similar[0]]
        features2 = similarity.features[most_similar[1]]
        
        for feature in SongSimilarity.WEIGHTS.keys():
            if feature == 'tempo':
                diff = abs(features1[feature] - features2[feature])
                sim = 1 / (1 + diff / 10)
            elif feature in ['centroid', 'rolloff']:
                diff = abs(features1[feature] - features2[feature])
                sim = 1 / (1 + diff / 1000)
            else:
                dist = np.linalg.norm(features1[feature] - features2[feature])
                sim = 1 / (1 + dist / 5)
            print(f"{feature}: {sim:.4f} (weight: {SongSimilarity.WEIGHTS[feature]})")

if __name__ == "__main__":
    main()