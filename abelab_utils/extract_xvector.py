# wavファイルのpathを投げると，192次元のSpeechBrain x-vectorが返ってくる
# https://speechbrain.readthedocs.io/en/latest/API/speechbrain.pretrained.interfaces.html#speechbrain.pretrained.interfaces.EncoderClassifier
# https://github.com/espnet/espnet/blob/af70425293e6c15fb56f83cced78afe17878b757/egs2/TEMPLATE/asr1/pyscripts/utils/extract_xvectors.py#L68-L111
# extractor = ExtractXvector()
# spembs = extractor(wav_path)

import torch
import numpy as np
import soundfile as sf

from speechbrain.dataio.preprocess import AudioNormalizer 
from speechbrain.pretrained import EncoderClassifier 

class ExtractXvector:
    def __init__(self):
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.classifier = EncoderClassifier.from_hparams( 
            source='speechbrain/spkrec-ecapa-voxceleb',
            run_opts={'device': self._device},
        ) 
        self.audio_norm = AudioNormalizer()

    def __call__(self, wav_path: str) -> np.ndarray:
        wav, sr = sf.read(wav_path)
        # Amp Normalization -1 ~ 1
        amax = np.amax(np.absolute(wav))
        wav = wav.astype(np.float32) / amax
        # Freq Norm
        wav = self.audio_norm(torch.from_numpy(wav), sr).to(self._device)
        # x-vector Embedding
        embeds = self.classifier.encode_batch(wav).detach().cpu().numpy()[0]
        # (1, 192) --> (192,)
        embeds = np.squeeze(embeds)

        return embeds
