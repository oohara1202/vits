# 書記素 or フルコンテキストラベルpathを投げると，音素・韻律記号が返ってくる
# G2P.from_grapheme(t)
# G2P.from_fcxlab(p)

import re
import pyopenjtalk
from jaconv import jaconv
from typing import Iterable, List, Optional, Union

class G2P:
    @classmethod
    def from_fcxlab(cls, lab_path: str, drop_unvoiced_vowels: bool = True) -> List[str]:
        labels = []

        with open(lab_path, 'r', encoding='utf8') as f:
            for line in f:
                _, _, l = line.rstrip().split(' ')
                labels.append(l)

        phones = cls._get_phonetic_prosodic_symbol(labels, drop_unvoiced_vowels)

        return phones

    @classmethod
    def from_grapheme(cls, text: str, drop_unvoiced_vowels: bool = True) -> List[str]:
        # テキストクリーナーjaconvで正規化
        text = jaconv.normalize(text)
        labels = pyopenjtalk.make_label(pyopenjtalk.run_frontend(text))
        phones = cls._get_phonetic_prosodic_symbol(labels, drop_unvoiced_vowels)
        
        return phones

    # https://espnet.github.io/espnet/_gen/espnet2.text.html#espnet2.text.phoneme_tokenizer.pyopenjtalk_g2p_prosody
    @classmethod
    def _get_phonetic_prosodic_symbol(cls, labels: List[str], drop_unvoiced_vowels: bool = True) -> List[str]:
        N = len(labels)

        phones = []
        for n in range(N):
            lab_curr = labels[n]

            # current phoneme
            p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)

            # deal unvoiced vowels as normal vowels
            if drop_unvoiced_vowels and p3 in "AEIOU":
                p3 = p3.lower()

            # deal with sil at the beginning and the end of text
            if p3 == "sil":
                assert n == 0 or n == N - 1
                if n == 0:
                    phones.append("^")
                elif n == N - 1:
                    # check question form or not
                    e3 = cls._numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                    if e3 == 0:
                        phones.append("$")
                    elif e3 == 1:
                        phones.append("?")
                continue
            elif p3 == "pau":
                phones.append("_")
                continue
            else:
                phones.append(p3)

            # accent type and position info (forward or backward)
            a1 = cls._numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
            a2 = cls._numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
            a3 = cls._numeric_feature_by_regex(r"\+(\d+)/", lab_curr)

            # number of mora in accent phrase
            f1 = cls._numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

            a2_next = cls._numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])
            # accent phrase border
            if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
                phones.append("#")
            # pitch falling
            elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
                phones.append("]")
            # pitch rising
            elif a2 == 1 and a2_next == 2:
                phones.append("[")

        return phones

    @staticmethod
    def _numeric_feature_by_regex(regex, s):
        match = re.search(regex, s)
        if match is None:
            return -50
        return int(match.group(1))
