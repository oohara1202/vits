from typing import Iterable, List, Optional, Union

from g2p import G2P

def text_clean(filelist: List[str]) -> List[str]:
    filelist_cleaned = list()
    for line in filelist:
        line = line.replace('\n', '')
        name, grapheme = line.split('|')

        # G2P適用
        symbols = ' '.join(G2P.from_grapheme(grapheme))

        # 再構成
        newline = f'{name}|{symbols}\n'
        filelist_cleaned.append(newline)
    
    return filelist_cleaned
