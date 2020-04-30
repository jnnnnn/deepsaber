import sys, os, pathlib
import json, pickle
import time
import torch
import numpy as np
from math import ceil
from pathlib import Path
from scipy import signal

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

DEFAULT_NOTE = {
    "_time": 0.0,
    "_cutDirection": 1,
    "_lineIndex": 1,
    "_lineLayer": 1,
    "_type": 0,
}


def main():
    input_folder = "c:/users/j/Downloads/songs/gen"
    for song_file in os.listdir(input_folder):
        mapify(os.path.join(input_folder, song_file))


def mapify(
    song_path,
    experiment_name="block_placement_ddc2/",
    checkpoint=130000,
    temperature=1.00,
    peak_threshold=0.5,
    bpm=128,
    output_folder="C:/Program Files (x86)/Steam/steamapps/common/Beat Saber/Beat Saber_Data/CustomLevels",
):
    opt = load_options(experiment_name, checkpoint)
    opt.generate_folder = output_folder
    model = prepare_model(opt)
    hop, features = extract_features(song_path, opt)

    song = torch.tensor(features).unsqueeze(0)

    level_folder = make_level_folder(
        song_path, output_folder, experiment_name, checkpoint, temperature
    )

    print("Generating level timings... (sorry I'm a bit slow)")
    peak_probs = generate_peak_probs(opt, model, song, temperature, features)
    notes = extract_notes(peak_probs, bpm, hop, opt, peak_threshold)
    make_level_from_notes({"Expert": notes}, song_path, level_folder)


def extract_notes(peak_probs, bpm, hop, opt, peak_threshold):
    print("Thresholding...")
    peaks = threshold_peaks(peak_probs, opt, peak_threshold)
    print(f"Generated {len(peaks)} peaks. Processing notes...")
    print("Processing notes...")
    times_real = [float(i * hop / opt.sampling_rate) for i in peaks]
    notes = [{**DEFAULT_NOTE, "_time": float(t * bpm / 60)} for t in times_real]
    print("Number of generated notes: ", len(notes))
    notes = np.array(notes)[
        np.where(np.diff([-1] + times_real) > constants.HUMAN_DELTA)[0]
    ].tolist()
    print("Number of generated notes (after pruning): ", len(notes))
    return notes


def load_options(experiment_name, checkpoint, cuda=True):
    """loading opt object from experiment's training"""
    with open(os.path.join("../training", experiment_name, "opt.json"), "r") as f:
        opt = json.load(f)

    # we assume we have 1 GPU in generating machine :P
    opt["cuda"] = cuda
    opt["gpu_ids"] = [0] if cuda else []
    opt["load_iter"] = checkpoint
    opt["checkpoint"] = checkpoint
    opt["experiment_name"] = experiment_name.split("/")[0]
    if "dropout" not in opt:  # for older experiments
        opt["dropout"] = 0.0

    # AttrDict is a bit error-prone but I can't be bothered cleaning it up
    class AttrDict:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    return AttrDict(**opt)


def prepare_model(opt):
    model = create_model(opt)
    model.setup()
    if "wavenet" in opt.model:
        model.receptive_field = (
            model.net.module.receptive_field if opt.cuda else model.net.receptive_field
        )
    else:
        model.receptive_field = 1
    model.load_networks(f"iter_{opt.load_iter}")

    return model


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


# python generate_stage2.py --cuda --song_path $song_path --json_file $json_file --experiment_name block_selection_new2 --checkpoint 2150000 --bpm 128 --temperature 1.00 --use_beam_search

# beep

if __name__ == "__main__":
    main()
