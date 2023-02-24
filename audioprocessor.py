import sys
import traceback
from pathlib import Path
from time import perf_counter as timer

import numpy as np
import torch

from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from toolbox.ui import UI
from toolbox.utterance import Utterance
from vocoder import inference as vocoder
from utils.default_models import ensure_default_models
import sounddevice as sd

class AudioProcessor():

    def __init__(self, audioPath, textPath):
        self.audioPath = audioPath
        self.textPath = textPath


    def getTxtFromFile(self, path):
        f = open(path, 'r')
        txt = (f.read())
        f.close()
        txt  = txt.replace('.', '.\n')
        return txt
    
    def init_encoder(self):
        encoder.load_model(Path("saved_models/default/encoder.pt"))

    def init_synthesizer(self):
        self.synthesizer = Synthesizer(Path("saved_models/default/synthesizer.pt"))
    
    def init_vocoder(self):
        vocoder.load_model(Path("saved_models/default/vocoder.pt"))   

    def synthesize(self):
        self.init_synthesizer()
        self.texts = self.getTxtFromFile(self.textPath)
        self.pre_processed_wav = encoder.preprocess_wav(self.audioPath)
        original_wav,sample_rate = librosa.load(self.audioPath)
        self.pre_processed_wav = encoder.preprocess_wav(original_wav,sample_rate)
        self.embed = encoder.embed_utterance(self.pre_processed_wav)
        self.embeds = [self.embed]*len(self.texts)
        specs = self.synthesizer.synthesize_spectrograms(self.texts, self.embeds)
        self.breaks = [spec.shape[1] for spec in specs]
        self.spec = np.concatenate(specs, axis=1)

    def vocode(self):
        self.init_vocoder()
        self.texts = self.getTxtFromFile(self.textPath)
        self.pre_processed_wav = encoder.preprocess_wav(self.audioPath)
        original_wav,sample_rate = librosa.load(self.audioPath)
        self.pre_processed_wav = encoder.preprocess_wav(original_wav,sample_rate)
        self.embed = encoder.embed_utterance(self.pre_processed_wav)
        self.embeds = [self.embed]*len(self.texts)
        specs = self.synthesizer.synthesize_spectrograms(self.texts, self.embeds)
        self.breaks = [spec.shape[1] for spec in specs]
        self.spec = np.concatenate(specs, axis=1)
        self.wav = vocoder.infer_waveform(self.spec)
        b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
        b_starts = np.concatenate(([0], b_ends[:-1]))
        self.wavs = [self.wav[start:end] for start, end, in zip(b_starts, b_ends)]
        breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
        self.wav = np.concatenate([i for w, b in zip(self.wavs, self.breaks) for i in (w, b)])
        self.wav = self.wav / np.abs(self.wav).max() * 0.97

        sd.stop()
        sd.play(self.wav,self.process_syn.sample_rate)



