import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import re

from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer 
from vocoder import inference as vocoder
from utils.default_models import ensure_default_models
from toolbox.utterance import Utterance
import sounddevice as sd

class helper:

    def __init__(self,audio_path,text_path):
        self.audio_path = audio_path
        self.text_path = text_path


    def getTxtFromFile(self, path):
        f = open(path, 'r')
        txt = (f.read())
        f.close()
        txt  = txt.replace('.', '.\n')
        return txt
    
    

    # def loadDefaultDataSet(self):
    #     ensure_default_models(Path("saved_models"))
    #     encoder.load_model(Path("saved_models/default/encoder.pt"))
    #     self.synthesizer = Synthesizer(Path("saved_models/default/synthesizer.pt"))
    #     vocoder.load_model(Path("saved_models/default/vocoder.pt"))

    

    def loadFromBrowser(self):
        self.wav = Synthesizer.load_preprocess_wav(self.audio_path)
        self.spec = Synthesizer.make_spectrogram(self.wav)
        encoder.load_model(Path("saved_models/default/encoder.pt"))
        self.encoder_wav = encoder.preprocess_wav(self.wav)
        self.embed, self.partial_embeds, _ = encoder.embed_utterance(self.encoder_wav, return_partials=True)
        utterance = Utterance("Deep-Stream", "Avishak", self.wav, self.spec, self.embed, self.partial_embeds, False)

    
    def synthesize(self):
        self.synthesizer = Synthesizer(Path("saved_models/default/synthesizer.pt"))
        self.texts = self.getTxtFromFile(self.text_path).split('\n')
        #Check new lines in the text
        self.embed = self.embed
        self.embeds = [self.embed]*len(self.texts)
        self.specs = self.synthesizer.synthesize_spectrograms(self.texts, self.embeds)
        self.breaks = [spec.shape[1] for spec in self.specs]
        self.spec = np.concatenate(self.specs, axis=1)
        self.current_generated = ("Avishak",self.spec,self.breaks,None)
    

    def vocoder(self):
        speaker_name, spec, breaks, _ = self.current_generated
        assert spec is not None
        vocoder.load_model(Path("saved_models/default/vocoder.pt"))
        self.wav = Synthesizer.griffin_lim(spec)

        #Check vocoder path and infer vocoder

        b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
        b_starts = np.concatenate(([0], b_ends[:-1]))
        wavs = [self.wav[start:end] for start, end, in zip(b_starts, b_ends)]
        breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
        self.wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])









    def preprocess(self):
        self.synthesize() or self.vocoder()
        # self.loadDefaultDataSet()
        # preprocessed_wav = encoder.preprocess_wav(audioPath)
        #     # - If the wav is already loaded:
        # original_wav, sampling_rate = librosa.load(str(audioPath))
        # preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
        # encoder.embed_utterance(np.zeros(encoder.sampling_rate))
        # embed = encoder.embed_utterance(preprocessed_wav)
        # text = self.getTxtFromFile(textPath)
        # texts = [text]
        # embeds = [embed]
        
        # specs = self.process_syn.synthesize_spectrograms(texts, embeds)
        # breaks = [spec.shape[1] for spec in specs]
        # spec = specs[0]

        # wav = vocoder.infer_waveform(spec)

        # generated_wav = vocoder.infer_waveform(spec)
        # b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
        # b_starts = np.concatenate(([0], b_ends[:-1]))

        # wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
        # breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
        # wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

        # wav = encoder.preprocess_wav(wav)
        # wav = wav / np.abs(wav).max() * 0.97

        # sd.stop()
        # sd.play(wav,self.process_syn.sample_rate)


        # self.pre_processed_wav = encoder.preprocess_wav(self.audio_path)
        # self.text= self.getTxtFromFile(self.text_path)
        # self.texts = [self.text]
        # original_wav,sample_rate = librosa.load(self.audio_path)
        # self.pre_processed_wav = encoder.preprocess_wav(original_wav,sample_rate)
        # self.embed = encoder.embed_utterance(self.pre_processed_wav)
        # self.embeds = [self.embed]
        # specs = self.synthesizer.synthesize_spectrograms(self.texts, self.embeds)
        # self.breaks = [spec.shape[1] for spec in specs]
        # self.spec = np.concatenate(specs, axis=1)
        # self.wav = vocoder.infer_waveform(self.spec)
        # b_ends = np.cumsum(np.array(self.breaks) * Synthesizer.hparams.hop_size)
        # b_starts = np.concatenate(([0], b_ends[:-1]))
        # self.wavs = [self.wav[start:end] for start, end, in zip(b_starts, b_ends)]
        # self.breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(self.breaks)
        # self.wav = np.concatenate([i for w, b in zip(self.wavs, self.breaks) for i in (w, b)])
        # self.wav = self.wav / np.abs(self.wav).max() * 0.97

        sd.stop()
        print(type(self.wav))
        filename = "cloned_audio.wav"
        sf.write(filename, self.wav, self.synthesizer.sample_rate)
        return True
        

        # generated_wav = np.pad(generated_wav, (0, self.process_syn.sample_rate), mode="constant")
        # generated_wav = encoder.preprocess_wav(generated_wav)
        # sd.stop()
        # sd.play(generated_wav, self.process_syn.sample_rate)


        # mel = np.concatenate(specs, axis=1)

        # no_action = lambda *args: None

        # vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)
        


        
        


