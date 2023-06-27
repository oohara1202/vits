""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.
'''
_pad        = '_'
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"


# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

# 以下日本語音素とESPnet2におけるNHK方式の韻律記号
# スペース区切りな点に注意
_vowel = 'a i u e o'
_unvoiced_vowel = 'A I U E O'
_consonant = 'b d f g h j k m n p r s t v w y z by dy gy hy ky my ny py ry ty ch sh ts'
_other = 'cl N'
_prosody = '^ $ ? _ # ] ['

# シンボルリストに追記
symbols = symbols + _vowel.split(' ') + _unvoiced_vowel.split(' ') + _consonant.split(' ') + _other.split(' ') + _prosody.split(' ')

# default (En)との重複を削除（圧倒的に横着）
# TODO: 言語ごとに切り替えたい
symbols = list(dict.fromkeys(symbols))

# Special symbol ids
SPACE_ID = symbols.index(" ")
