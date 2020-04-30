python generate_stage1.py --cuda --song_path $song_path --experiment_name block_placement_ddc2 --checkpoint 130000 --bpm 128 --peak_threshold ${peak_threshold} --temperature 1.0





import sys, os, pathlib
import argparse
import json, pickle
import time
import torch
import numpy as np
from math import ceil
from pathlib import Path
from scipy import signal


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXTRACT_DIR = os.path.join(DATA_DIR, 'extracted_data')
Path(DATA_DIR).mkdir(exist_ok=True)
Path(EXTRACT_DIR).mkdir(exist_ok=True)

sys.path.append(ROOT_DIR)
from models import create_model
import models.constants as constants
from scripts.generation.level_generation_utils import extract_features, make_level_from_notes, get_notes_from_stepmania_file

parser = argparse.ArgumentParser(description='Generate Beat Saber level from song')
parser.add_argument('--song_path', type=str)
parser.add_argument('--experiment_name', type=str)
parser.add_argument('--peak_threshold', type=float, default=0.0148)
parser.add_argument('--checkpoint', type=str, default="latest")
parser.add_argument('--temperature', type=float, default=1.00)
parser.add_argument('--bpm', type=float, default=None)
parser.add_argument('--cuda', action="store_true")

args = parser.parse_args()

temperature=1.00
song_path=args.song_path

song_name = Path(song_path).stem

DEFAULT_NOTE = {"_time": 0.0, "_cutDirection":1, "_lineIndex":1, "_lineLayer":1, "_type":0}

def main(song_path, experiment_name = "block_placement_ddc2/", checkpoint = 130000, temperature = 1.00, peak_threshold=0.0148, bpm=128):
    opt = load_options(experiment_name, checkpoint)
    model = prepare_model(opt)
    hop, features = extract_features(song_path, args, opt)

    song = torch.tensor(features).unsqueeze(0)

    print("Generating level timings... (sorry I'm a bit slow)")
    peak_probs = generate_peak_probs(opt, model, song, temperature, features)
    print("Thresholding...")
    peaks = threshold_peaks(peak_probs, opt, peak_threshold)
    print(f"Generated {len(peaks)} peaks. Processing notes...")
    print("Processing notes...")
    times_real = [float(i*hop/opt.sampling_rate) for i in peaks]
    notes = [{**DEFAULT_NOTE, "_time":float(t*bpm/60)} for t in times_real]
    print("Number of generated notes: ", len(notes))
    notes = np.array(notes)[np.where(np.diff([-1]+times_real) > constants.HUMAN_DELTA)[0]].tolist()
    print("Number of generated notes (after pruning): ", len(notes))
    json_file = make_level_from_notes(notes, bpm, song_name, args)

    print(json_file)


def load_options(experiment_name, checkpoint, cuda=True):
    '''loading opt object from experiment's training'''
    opt = json.loads(open("../training/"+experiment_name+"opt.json","r").read())
    # we assume we have 1 GPU in generating machine :P
    opt["cuda"] = cuda
    opt["gpu_ids"] = [0] if cuda else []
    opt["load_iter"] = int(checkpoint)
    opt["experiment_name"] = args.experiment_name.split("/")[0]
    if "dropout" not in opt: #for older experiments
        opt["dropout"] = 0.0
    
    assert opt.binarized
    
    # AttrDict is a bit error-prone but I can't be bothered cleaning it up
    class AttrDict:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    return AttrDict(**opt)

def prepare_model(opt):
    model = create_model(opt)
    model.setup()
    if 'wavenet' in opt.model:
        model.receptive_field = model.net.module.receptive_field if opt.cuda else model.net.receptive_field
    else:
        model.receptive_field = 1
    model.load_networks("iter_"+opt.load_iter)

    return model

def generate_peak_probs(opt, model, song, temperature, features):
    if opt.concat_outputs: #whether to concatenate the generated outputs as new inputs (AUTOREGRESSIVE)
        # first_samples basically works as a padding, for the first few outputs, which don't have any "past part" of the song to look at.
        first_samples = torch.full((1,opt.output_channels,model.receptive_field//2),constants.START_STATE)
        output,peak_probs = model.net.module.generate(song.size(-1)-opt.time_shifts+1,song,time_shifts=opt.time_shifts,temperature=temperature,first_samples=first_samples)
        peak_probs = np.array(peak_probs)
    else: # NOT-AUTOREGRESSIVE (we keep it separate like this, because some models have both)
        peak_probs = model.generate(features)[0,:,-1].cpu().detach().numpy()
    return peak_probs

def threshold_peaks(peak_probs, opt, peak_threshold):
    window = signal.hamming(ceil(constants.HUMAN_DELTA/opt.step_size))
    smoothed_peaks = np.convolve(peak_probs,window,mode='same')
    thresholded_peaks = smoothed_peaks*(smoothed_peaks>peak_threshold)
    peaks = signal.find_peaks(thresholded_peaks)[0]
    return peaks





























python generate_stage2.py --cuda --song_path $song_path --json_file $json_file --experiment_name block_selection_new2 --checkpoint 2150000 --bpm 128 --temperature 1.00 --use_beam_search



[console]::beep(800,700)
[console]::beep(700,700)
[console]::beep(600,700)
