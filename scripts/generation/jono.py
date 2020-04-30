
MYPATHS = dict(
    InputMP3s = "c:/users/j/Downloads/songs/gen",
    OutputBeatSaberLevels = "C:/Program Files (x86)/Steam/steamapps/common/Beat Saber/Beat Saber_Data/CustomLevels"
)


import sys, os, pathlib
import json, pickle
import time
import torch
import numpy as np
from math import ceil
from pathlib import Path
from scipy import signal
import logging
import tqdm

logging.basicConfig(format='%(asctime)-15s %(message)s')
logger = logging.getLogger()

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
EXTRACT_DIR = os.path.join(DATA_DIR, "extracted_data")
Path(DATA_DIR).mkdir(exist_ok=True)
Path(EXTRACT_DIR).mkdir(exist_ok=True)
sys.path.append(ROOT_DIR)
from models import create_model
import models.constants as constants
from scripts.generation.level_generation_utils import (
    extract_features,
    make_level_from_notes,
)


from scripts.data_processing.state_space_functions import stage_two_states_to_json_notes

DEFAULT_NOTE = {
    "_time": 0.0,
    "_cutDirection": 1,
    "_lineIndex": 1,
    "_lineLayer": 1,
    "_type": 0,
}


def main():
    input_folder = MYPATHS['InputMP3s']
    for song_file in tqdm.tqdm(os.listdir(input_folder)):
        logger.info(f"\n\n\nmapping {song_file}")
        mapify(os.path.join(input_folder, song_file))
        torch.cuda.empty_cache()  # avoid RuntimeError: CUDA out of memory after several songs


def model1():
    opt = load_options("block_placement_ddc2", 130000)
    model = create_model(opt)
    model.setup()
    if "wavenet" in opt.model:
        model.receptive_field = (
            model.net.module.receptive_field if opt.cuda else model.net.receptive_field
        )
    else:
        model.receptive_field = 1
    model.load_networks(f"iter_{opt.load_iter}")
    return opt, model


def model2():
    opt = load_options("block_selection_new2", 2150000)
    model = create_model(opt)
    model.setup()
    model.load_networks(f"iter_{opt.checkpoint}")
    return opt, model


def mapify(
    song_path,
    temperature=1.00,
    peak_threshold=0.5,
    bpm=128,
):
    opt, model = model1()
    hop, features = extract_features(song_path, opt)

    song = torch.tensor(features).unsqueeze(0)

    level_folder = make_level_folder(song_path, MYPATHS['OutputBeatSaberLevels'], temperature)

    logger.info("Generating level timings... (sorry I'm a bit slow)")
    peak_probs = generate_peak_probs(opt, model, song, temperature, features)
    difficulties = {
        "Easy": linspace_threshold(peak_probs, bpm, hop, opt, notes_per_sec=1.0),
        "Normal": linspace_threshold(peak_probs, bpm, hop, opt, notes_per_sec=1.5),
        "Hard": linspace_threshold(peak_probs, bpm, hop, opt, notes_per_sec=2.0),
        "Expert": linspace_threshold(peak_probs, bpm, hop, opt, notes_per_sec=2.5),
    }
    make_level_from_notes(difficulties, song_path, level_folder)

    # stage 2
    opt, model = model2()
    with open("../../data/statespace/sorted_states.pkl", "rb") as f:
        unique_states = pickle.load(f)
    hop, features = extract_features(song_path, opt)

    for diffi, notes in difficulties.items():
        logger.info(f"Generating state sequence for {diffi}")
        state_times, generated_sequence = model.generate(
            features,
            os.path.join(level_folder, f"{diffi}.dat"),
            bpm,
            unique_states,
            temperature=temperature,
            use_beam_search=True,
            generate_full_song=False,
        )
        times_real = [t * 60 / bpm for t in state_times]
        notes2 = stage_two_states_to_json_notes(
            generated_sequence,
            state_times,
            bpm,
            hop,
            opt.sampling_rate,
            state_rank=unique_states,
        )
        difficulties[diffi] = notes2

    make_level_from_notes(difficulties, song_path, level_folder)


def linspace_threshold(peak_probs, bpm, hop, opt, notes_per_sec=2):
    """try lots of thresholds and use the one with the closest notes per second rate"""
    best_notes = []
    best_difference = 999999
    for peak_threshold in np.geomspace(1, 0.0001, num=100):
        notes = extract_notes(peak_probs, bpm, hop, opt, peak_threshold)
        if len(notes) > 2:
            total_seconds = notes[-1]["_time"] - notes[0]["_time"]
            nps = len(notes) / total_seconds
        else:
            nps = 0
        difference = abs(notes_per_sec - nps)
        if difference < best_difference:
            best_notes = notes
            best_difference = difference
            best = f"Threshold {peak_threshold:.5f} gave {nps:.1f} notes per second for a total of {len(notes)}"
    logger.info(best)
    return best_notes


def binary_search_threshold(
    peak_probs, bpm, hop, opt, notes_per_sec=2, allowable_error=0.2
):
    """binary search peak_threshold until we get an acceptable notes per second rate
    
    After testing, I decided not to use this appraoch as the search space is apparently not monotonic
    """
    min_threshold = 0.0001
    max_threshold = 0.9999
    while True:
        peak_threshold = (max_threshold + min_threshold) / 2
        notes = extract_notes(peak_probs, bpm, hop, opt, peak_threshold)
        if len(notes) > 2:
            total_seconds = notes[-1]["_time"] - notes[0]["_time"]
            nps = len(notes) / total_seconds
        else:
            nps = 0
        logger.info(f"Threshold {peak_threshold} gives {nps} notes per sec")
        if abs(max_threshold - min_threshold) < 0.000001:
            return []
        elif nps > notes_per_sec + allowable_error:
            min_threshold = peak_threshold
        elif nps < notes_per_sec - allowable_error:
            max_threshold = peak_threshold
        else:
            return notes


def extract_notes(peak_probs, bpm, hop, opt, peak_threshold):
    peaks = threshold_peaks(peak_probs, opt, peak_threshold)
    times_real = [float(i * hop / opt.sampling_rate) for i in peaks]
    notes = [{**DEFAULT_NOTE, "_time": float(t * bpm / 60)} for t in times_real]
    notes = np.array(notes)[
        np.where(np.diff([-1] + times_real) > constants.HUMAN_DELTA)[0]
    ].tolist()
    return notes


def load_options(experiment_name, checkpoint, cuda=True):
    """loading opt object from experiment's training"""
    with open(os.path.join("..", "training", experiment_name, "opt.json"), "r") as f:
        opt = json.load(f)

    # we assume we have 1 GPU in generating machine :P
    opt["cuda"] = cuda
    opt["gpu_ids"] = [0] if cuda else []
    opt["load_iter"] = checkpoint
    opt["checkpoint"] = checkpoint
    opt["experiment_name"] = experiment_name
    if "dropout" not in opt:  # for older experiments
        opt["dropout"] = 0.0

    # stage 2
    opt["batch_size"] = 1
    opt["beam_size"] = 20
    opt["n_best"] = 1
    # opt["using_bpm_time_division"] = True
    opt["continue_train"] = False

    # AttrDict is a bit ugly but I can't be bothered cleaning it up
    class AttrDict:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    return AttrDict(**opt)


def generate_peak_probs(opt, model, song, temperature, features):
    if (
        opt.concat_outputs
    ):  # whether to concatenate the generated outputs as new inputs (AUTOREGRESSIVE)
        # first_samples basically works as a padding, for the first few outputs, which don't have any "past part" of the song to look at.
        first_samples = torch.full(
            (1, opt.output_channels, model.receptive_field // 2), constants.START_STATE
        )
        _output, peak_probs = model.net.module.generate(
            song.size(-1) - opt.time_shifts + 1,
            song,
            time_shifts=opt.time_shifts,
            temperature=temperature,
            first_samples=first_samples,
        )
        peak_probs = np.array(peak_probs)
    else:  # NOT-AUTOREGRESSIVE (we keep it separate like this, because some models have both)
        peak_probs = model.generate(features)[0, :, -1].cpu().detach().numpy()
    return peak_probs


def threshold_peaks(peak_probs, opt, peak_threshold):
    window = signal.hamming(ceil(constants.HUMAN_DELTA / opt.step_size))
    smoothed_peaks = np.convolve(peak_probs, window, mode="same")
    thresholded_peaks = smoothed_peaks * (smoothed_peaks > peak_threshold)
    peaks = signal.find_peaks(thresholded_peaks)[0]
    return peaks


def make_level_folder(song_path, output_folder, *args):
    level_folder = os.path.abspath(
        output_folder + "/" + "_".join([Path(song_path).stem, *[str(a) for a in args]])
    )
    if not os.path.exists(level_folder):
        os.makedirs(level_folder)
    return level_folder


# beep

if __name__ == "__main__":
    main()
