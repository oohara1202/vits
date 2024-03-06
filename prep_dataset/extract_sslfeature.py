# wavファイルのpathを投げると，[layer(13), frame(any), feature(768)]のSSLモデル特徴量が返ってくる
# https://huggingface.co/rinna/japanese-hubert-base
# extractor = ExtractSSLModelFeature()
# embeds = extractor(wav_path)

import torch
import numpy as np
import soundfile as sf

from speechbrain.dataio.preprocess import AudioNormalizer 
from transformers import HubertModel

class ExtractSSLModelFeature:
    def __init__(self):
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.model = HubertModel.from_pretrained('rinna/japanese-hubert-base')
        self.model.to(self._device)
        self.model.eval()

        self.audio_norm = AudioNormalizer()  # for resampling to 16kHz

    def __call__(self, wav_path: str) -> torch.Tensor:
        wav, sr = sf.read(wav_path)
        # Amp Normalization -1 ~ 1
        amax = np.amax(np.absolute(wav))
        wav = wav.astype(np.float32) / amax
        # Freq Norm
        wav = self.audio_norm(torch.from_numpy(wav), sr).unsqueeze(0).to(self._device)
        # Feature Extraction
        outputs = self.model(wav, output_hidden_states=True)

        outputs = torch.stack(outputs.get('hidden_states')).squeeze().cpu()  # [layer(13), frame(any), feature(768)]

        return outputs
