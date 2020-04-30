import os
import json
import subprocess
from shutil import copyfile
from pathlib import Path
import numpy as np

import librosa
from scripts.feature_extraction.feature_extraction import (
    extract_features_hybrid,
    extract_features_mel,
    extract_features_hybrid_beat_synced,
    extract_features_multi_mel,
)


def extract_features(song_path, opt, bpm=128):
    y_wav, sr = librosa.load(song_path, sr=opt.sampling_rate)

    # useful quantities
    feature_name = opt.feature_name
    feature_size = opt.feature_size
    sampling_rate = opt.sampling_rate
    beat_subdivision = opt.beat_subdivision
    try:
        step_size = opt.step_size
        using_bpm_time_division = opt.using_bpm_time_division
    except:  # older model
        using_bpm_time_division = True

    sr = sampling_rate
    beat_duration = 60 / bpm  # beat duration in seconds
    beat_duration_samples = int(60 * sr / bpm)  # beat duration in samples
    if using_bpm_time_division:
        # duration of one time step in samples:
        hop = int(beat_duration_samples * 1 / beat_subdivision)
        step_size = beat_duration / beat_subdivision  # in seconds
    else:
        beat_subdivision = 1 / (step_size * bpm / 60)
        hop = int(step_size * sr)

    # get feature
    if feature_name == "chroma":
        if using_bpm_time_division:
            state_times = np.arange(0, y_wav.shape[0] / sr, step=step_size)
            features = extract_features_hybrid_beat_synced(
                y_wav, sr, state_times, bpm, beat_discretization=1 / beat_subdivision
            )
        else:
            features = extract_features_hybrid(y_wav, sr, hop)
    elif feature_name == "mel":
        if using_bpm_time_division:
            raise NotImplementedError(
                "Mel features with beat synced times not implemented, but trivial TODO"
            )
        else:
            features = extract_features_mel(y_wav, sr, hop, mel_dim=feature_size)
    elif feature_name == "multi_mel":
        if using_bpm_time_division:
            raise NotImplementedError(
                "Mel features with beat synced times not implemented, but trivial TODO"
            )
        else:
            features = extract_features_multi_mel(
                y_wav,
                sr=sampling_rate,
                hop=hop,
                nffts=[1024, 2048, 4096],
                mel_dim=feature_size,
            )

    return hop, features


def make_difficulty(difficulty: str):
    return {
        "_difficulty": difficulty,
        "_difficultyRank": 7,
        "_beatmapFilename": f"{difficulty}.dat",
        "_noteJumpMovementSpeed": 10,
        "_noteJumpStartBeatOffset": 0,
        "_customData": {
            "_difficultyLabel": "",
            "_editorOffset": 0,
            "_editorOldOffset": 0,
            "_warnings": [],
            "_information": [],
            "_suggestions": [],
            "_requirements": [],
        },
    }


def make_level_from_notes(
    difficulties, song_path, level_folder,  # difficulties: {str: [note]},
):
    song_name = Path(song_path).stem

    info_json = {
        "_version": "2.0.0",
        "_songName": song_name,
        "_songSubName": "",
        "_songAuthorName": "DeepSaber",
        "_levelAuthorName": "DeepSaber",
        "_beatsPerMinute": 128,
        "_songTimeOffset": 0,
        "_shuffle": 0,
        "_shufflePeriod": 0.5,
        "_previewStartTime": 12,
        "_previewDuration": 10,
        "_songFilename": "song.ogg",
        "_coverImageFilename": "cover.jpg",
        "_environmentName": "NiceEnvironment",
        "_customData": {
            "_contributors": [],
            "_customEnvironment": "",
            "_customEnvironmentHash": "",
        },
        "_difficultyBeatmapSets": [
            {
                "_beatmapCharacteristicName": "Standard",
                "_difficultyBeatmaps": [make_difficulty(d) for d in difficulties],
            }
        ],
    }

    for d in difficulties:
        with open(f"{level_folder}/{d}.dat", "w") as f:
            f.write(
                json.dumps(
                    f, {"_events": [], "_notes": difficulties[d], "_obstacles": [],},
                )
            )

    with open(level_folder + "/info.dat", "w") as f:
        f.write(json.dumps(f, info_json))

    # copyfile(logo_path, level_folder + "/cover.jpg")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            song_path,
            "-c:a",
            "libvorbis",
            "-q:a",
            "4",
            level_folder + "/song.ogg",
        ]
    )
