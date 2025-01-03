#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('python --version')


# In[4]:


get_ipython().system('pip list')


# In[ ]:


get_ipython().system('pip install scikit-learn==1.0.2')


# In[ ]:


get_ipython().system('pip install numpy==1.23.5')


# In[ ]:


get_ipython().system('pip install tensorflow==2.11.0')


# In[ ]:


get_ipython().system('pip install tensorflow_addons==0.19.0')


# In[ ]:


get_ipython().system('pip install keras==2.11.0')


# In[ ]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install kobert-transformers')
get_ipython().system('pip install sentencepiece')
get_ipython().system('pip install transformers torch sentencepiece')
get_ipython().system('pip install torch')
get_ipython().system('pip install ipywidgets')
get_ipython().system('pip install tf-keras')
# !pip install tensorflow_addons
get_ipython().system('pip install pandas')

get_ipython().system('pip install typeguard==2.13.3')

get_ipython().system('pip install tensorflow==2.11.0')
get_ipython().system('pip install tensorflow_addons==0.19.0')
get_ipython().system('pip install numpy==1.23.5')


# In[ ]:


get_ipython().system('pip install pandas')


# In[ ]:


import keras


# In[ ]:


import tensorflow as tf
import numpy as np
import pandas as pd
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sentencepiece as spm
import urllib.request
import zipfile
import os
import logging
import unicodedata
from shutil import copyfile

from transformers import PreTrainedTokenizer

import logging
import os
import unicodedata
from shutil import copyfile

from transformers import PreTrainedTokenizer
import tensorflow_addons


##kobert
import subprocess
installed_packages = subprocess.run(['pip', 'list'], capture_output=True, text=True)
for line in installed_packages.stdout.split('\n'):
    if 'kobert' in line:
        print(line)

import kobert_transformers
from kobert_transformers import get_tokenizer
from kobert_transformers.utils import get_tokenizer
from kobert_transformers import get_kobert_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

import tensorflow as tf
from transformers import TFBertModel


# In[ ]:


train = pd.read_table('ratings_train.txt', sep='\t')
test = pd.read_table("ratings_test.txt",sep='\t')


# In[ ]:


train[1:50]


# In[ ]:


import logging
import os
import unicodedata
from shutil import copyfile

from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer_78b3253a26.model",
                     "vocab_txt": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "monologg/kobert": "https://s3.amazonaws.com/models.huggingface.co/bert/monologg/kobert/tokenizer_78b3253a26.model",
        "monologg/kobert-lm": "https://s3.amazonaws.com/models.huggingface.co/bert/monologg/kobert-lm/tokenizer_78b3253a26.model",
        "monologg/distilkobert": "https://s3.amazonaws.com/models.huggingface.co/bert/monologg/distilkobert/tokenizer_78b3253a26.model"
    },
    "vocab_txt": {
        "monologg/kobert": "https://s3.amazonaws.com/models.huggingface.co/bert/monologg/kobert/vocab.txt",
        "monologg/kobert-lm": "https://s3.amazonaws.com/models.huggingface.co/bert/monologg/kobert-lm/vocab.txt",
        "monologg/distilkobert": "https://s3.amazonaws.com/models.huggingface.co/bert/monologg/distilkobert/vocab.txt"
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "monologg/kobert": 512,
    "monologg/kobert-lm": 512,
    "monologg/distilkobert": 512
}

PRETRAINED_INIT_CONFIGURATION = {
    "monologg/kobert": {"do_lower_case": False},
    "monologg/kobert-lm": {"do_lower_case": False},
    "monologg/distilkobert": {"do_lower_case": False}
}

SPIECE_UNDERLINE = u'▁'


class KoBertTokenizer(PreTrainedTokenizer):
    """
        SentencePiece based tokenizer. Peculiarities:
            - requires `SentencePiece `_
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
            self,
            vocab_file,
            vocab_txt,
            do_lower_case=False,
            remove_space=True,
            keep_accents=False,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            **kwargs):
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs
        )

        # Build vocab
        self.token2idx = dict()
        self.idx2token = []
        with open(vocab_txt, 'r', encoding='utf-8') as f:
            for idx, token in enumerate(f):
                token = token.strip()
                self.token2idx[token] = idx
                self.idx2token.append(token)

        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning("You need to install SentencePiece to use KoBertTokenizer: https://github.com/google/sentencepiece"
                           "pip install sentencepiece")

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file
        self.vocab_txt = vocab_txt

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

    @property
    def vocab_size(self):
        return len(self.idx2token)

    def get_vocab(self):
        return dict(self.token2idx, **self.added_tokens_encoder)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning("You need to install SentencePiece to use KoBertTokenizer: https://github.com/google/sentencepiece"
                           "pip install sentencepiece")
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

    def preprocess_text(self, inputs):
        if self.remove_space:
            outputs = " ".join(inputs.strip().split())
        else:
            outputs = inputs
        outputs = outputs.replace("``", '"').replace("''", '"')

        if not self.keep_accents:
            outputs = unicodedata.normalize('NFKD', outputs)
            outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
        if self.do_lower_case:
            outputs = outputs.lower()

        return outputs

    def _tokenize(self, text, return_unicode=True, sample=False):
        """ Tokenize a string. """
        text = self.preprocess_text(text)

        if not sample:
            pieces = self.sp_model.EncodeAsPieces(text)
        else:
            pieces = self.sp_model.SampleEncodeAsPieces(text, 64, 0.1)
        new_pieces = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == str(",") and piece[-2].isdigit():
                cur_pieces = self.sp_model.EncodeAsPieces(piece[:-1].replace(SPIECE_UNDERLINE, ""))
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)

        return new_pieces

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.token2idx.get(token, self.token2idx[self.unk_token])

    def _convert_id_to_token(self, index, return_unicode=True):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        return self.idx2token[index]

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A KoBERT sequence has the following format:
            single sequence: [CLS] X [SEP]
            pair of sequences: [CLS] A [SEP] B [SEP]
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.
        Args:
            token_ids_0: list of ids (must not contain special tokens)
            token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                for sequence pairs
            already_has_special_tokens: (default False) Set to True if the token list is already formated with
                special tokens for the model
        Returns:
            A list of integers in the range [0, 1]: 0 for a special token, 1 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A KoBERT sequence pair mask has the following format:
        0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence
        if token_ids_1 is None, only returns the first portion of the mask (0's).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory):
        """ Save the sentencepiece vocabulary (copy original file) and special tokens file
            to a directory.
        """
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return

        # 1. Save sentencepiece model
        out_vocab_model = os.path.join(save_directory, VOCAB_FILES_NAMES["vocab_file"])

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_model):
            copyfile(self.vocab_file, out_vocab_model)

        # 2. Save vocab.txt
        index = 0
        out_vocab_txt = os.path.join(save_directory, VOCAB_FILES_NAMES["vocab_txt"])
        with open(out_vocab_txt, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.token2idx.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(out_vocab_txt)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1

        return out_vocab_model, out_vocab_txt


# In[ ]:


from transformers import BertModel, BertTokenizer
tokenizer = get_tokenizer()
model = BertModel.from_pretrained('monologg/kobert')


# In[ ]:


print(tokenizer.encode("보는내내 그대로 들어맞는 예측 카리스마 없는 악역"))


# In[ ]:


print(tokenizer.tokenize("보는내내 그대로 들어맞는 예측 카리스마 없는 악역"))


# In[ ]:


print(tokenizer.tokenize("전율을 일으키는 영화. 다시 보고싶은 영화"))


# In[ ]:


# 세그멘트 인풋
print([0]*64)


# In[ ]:


# 마스크 인풋
valid_num = len(tokenizer.encode("전율을 일으키는 영화. 다시 보고싶은 영화"))
print(valid_num * [1] + (64 - valid_num) * [0])


# In[ ]:


def convert_data(data_df):
    global tokenizer

    SEQ_LEN = 64 #SEQ_LEN : 버트에 들어갈 인풋의 길이

    tokens, masks, segments, targets = [], [], [], []

    for i in tqdm(range(len(data_df))):
        # token : 문장을 토큰화함
        token = tokenizer.encode(data_df[DATA_COLUMN][i], truncation=True, padding='max_length', max_length=SEQ_LEN)

        # 마스크는 토큰화한 문장에서 패딩이 아닌 부분은 1, 패딩인 부분은 0으로 통일
        num_zeros = token.count(0)
        mask = [1]*(SEQ_LEN-num_zeros) + [0]*num_zeros

        # 문장의 전후관계를 구분해주는 세그먼트는 문장이 1개밖에 없으므로 모두 0
        segment = [0]*SEQ_LEN

        # 버트 인풋으로 들어가는 token, mask, segment를 tokens, segments에 각각 저장
        tokens.append(token)
        masks.append(mask)
        segments.append(segment)

        # 정답(긍정 : 1 부정 0)을 targets 변수에 저장해 줌
        targets.append(data_df[LABEL_COLUMN][i])

    # tokens, masks, segments, 정답 변수 targets를 numpy array로 지정
    tokens = np.array(tokens)
    masks = np.array(masks)
    segments = np.array(segments)
    targets = np.array(targets)

    return [tokens, masks, segments], targets

# 위에 정의한 convert_data 함수를 불러오는 함수를 정의
def load_data(pandas_dataframe):
    data_df = pandas_dataframe
    data_df[DATA_COLUMN] = data_df[DATA_COLUMN].astype(str)
    data_df[LABEL_COLUMN] = data_df[LABEL_COLUMN].astype(int)
    data_x, data_y = convert_data(data_df)
    return data_x, data_y

SEQ_LEN = 64
BATCH_SIZE = 32
# 긍부정 문장을 포함하고 있는 칼럼
DATA_COLUMN = "document"
# 긍정인지 부정인지를 (1=긍정,0=부정) 포함하고 있는 칼럼
LABEL_COLUMN = "label"

# train 데이터를 버트 인풋에 맞게 변환
train_x, train_y = load_data(train)


# In[ ]:


# 훈련 성능을 검증한 test 데이터를 버트 인풋에 맞게 변환
test_x, test_y = load_data(test)


# ### 버트를 활용하여 감성분석 모델 만들기

# In[ ]:


model = TFBertModel.from_pretrained("monologg/kobert", from_pt=True)
# 토큰 인풋, 마스크 인풋, 세그먼트 인풋 정의
token_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_word_ids')
mask_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_masks')
segment_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_segment')
# 인풋이 [토큰, 마스크, 세그먼트]인 모델 정의
bert_outputs = model([token_inputs, mask_inputs, segment_inputs])


# In[ ]:


bert_outputs


# In[ ]:


bert_outputs=bert_outputs[1]


# In[ ]:


import tensorflow_addons as tfa

opt = tfa.optimizers.RectifiedAdam(learning_rate=5.0e-5, total_steps = 2344*2, warmup_proportion=0.1, min_lr=1e-5, epsilon=1e-08, clipnorm=1.0)


# In[ ]:


sentiment_drop = tf.keras.layers.Dropout(0.5)(bert_outputs)
sentiment_first = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(sentiment_drop)
sentiment_model = tf.keras.Model([token_inputs, mask_inputs, segment_inputs], sentiment_first)
sentiment_model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])


# In[ ]:


sentiment_model.summary()


# ### 모델 훈련 및 성능 검증

# In[ ]:


sentiment_model.fit(train_x, train_y, epochs=2, shuffle=True, batch_size=64, validation_data=(test_x,test_y))


# 훈련모델 저장하기 및 불러오기

# In[ ]:


## 훈련모델 저장하기
# sentiment_model.save('sentiment_model.h5')


# In[1]:


from tensorflow.keras.models import load_model
from transformers import TFBertModel
import tensorflow_addons as tfa  # RectifiedAdam을 사용하기 위한 import

# 모델 불러오기 시 custom_objects에 RectifiedAdam과 TFBertModel을 추가
sentiment_model = load_model('sentiment_model.h5', custom_objects={
    'TFBertModel': TFBertModel,
    'Addons>RectifiedAdam': tfa.optimizers.RectifiedAdam
})


# In[ ]:


sentiment_model.summary()


# ### 실행

# In[2]:


def predict_convert_data(data_df):
    global tokenizer
    tokens, masks, segments = [], [], []

    for i in tqdm(range(len(data_df))):

        token = tokenizer.encode(data_df[DATA_COLUMN][i], max_length=SEQ_LEN, truncation=True, padding='max_length')
        num_zeros = token.count(0)
        mask = [1]*(SEQ_LEN-num_zeros) + [0]*num_zeros
        segment = [0]*SEQ_LEN

        tokens.append(token)
        segments.append(segment)
        masks.append(mask)

    tokens = np.array(tokens)
    masks = np.array(masks)
    segments = np.array(segments)
    return [tokens, masks, segments]

# 위에 정의한 convert_data 함수를 불러오는 함수를 정의
def predict_load_data(pandas_dataframe):
    data_df = pandas_dataframe
    data_df[DATA_COLUMN] = data_df[DATA_COLUMN].astype(str)
    data_x = predict_convert_data(data_df)
    return data_x


# In[3]:


test_set = predict_load_data(test)


# In[ ]:


test_set


# ### kobert model sentiment_model로 예측

# In[ ]:


preds = sentiment_model.predict(test_set)


# In[ ]:


# 부정이면 0, 긍정이면 1 출력
preds


# In[ ]:


from sklearn.metrics import classification_report
y_true = test['label']
# F1 Score 확인
print(classification_report(y_true, np.round(preds,0)))


# In[ ]:


import logging
tf.get_logger().setLevel(logging.ERROR)


# ### 실제 분류

# In[ ]:


def evaluation_predict():
    # 사용자로부터 텍스트 입력받기
    sentence = input("일기를 입력하세요: ")

    # 입력된 텍스트를 감성 분석 모델에 전달
    data_x = sentence_convert_data(sentence)
    predict = sentiment_model.predict(data_x)
    predict_value = np.ravel(predict)
    predict_answer = np.round(predict_value, 0).item()

    # 반환할 결과 문자열 초기화
    result = ""

    if predict_answer == 0:
        # 범위에 따라 극복 단어를 선택
        if predict_value >= 0.50 and predict_value < 0.60:
            emotion = "극복"
        elif predict_value >= 0.60 and predict_value < 0.70:
            emotion = "회복"
        elif predict_value >= 0.70 and predict_value < 0.80:
            emotion = "인내"
        elif predict_value >= 0.80 and predict_value < 0.85:
            emotion = "성장"
        elif predict_value >= 0.85 and predict_value < 0.90:
            emotion = "도전"
        elif predict_value >= 0.90 and predict_value < 0.95:
            emotion = "희망"
        elif predict_value >= 0.95 and predict_value <= 1.00:
            emotion = "용기"
        else:
            emotion = "끈기"

        # 문자열 결과 생성
        result = "(부정 확률 : %.2f) %s을(를) 필요로 하는 일기 내용" % (1 - predict_value, emotion)

    elif predict_answer == 1:
        # 긍정일 때는 기존 감정 단어로 출력
        if predict_value >= 0.50 and predict_value < 0.60:
            emotion = "희망찬"
        elif predict_value >= 0.60 and predict_value < 0.70:
            emotion = "따스한"
        elif predict_value >= 0.70 and predict_value < 0.80:
            emotion = "활기찬"
        elif predict_value >= 0.80 and predict_value < 0.85:
            emotion = "기쁨 가득한"
        elif predict_value >= 0.85 and predict_value < 0.90:
            emotion = "상쾌한"
        elif predict_value >= 0.90 and predict_value < 0.95:
            emotion = "환한"
        elif predict_value >= 0.95 and predict_value <= 1.00:
            emotion = "행복한"
        else:
            emotion = "즐거운"

        # 문자열 결과 생성
        result = "(긍정 확률 : %.2f) %s 일기의 내용" % (predict_value, emotion)

    return result


# In[ ]:


def sentence_convert_data(sentence):
    token = tokenizer.encode(sentence, max_length=SEQ_LEN, truncation=True, padding='max_length')
    num_zeros = token.count(0)
    mask = [1]*(SEQ_LEN-num_zeros) + [0]*num_zeros
    segment = [0]*SEQ_LEN

    tokens = np.array([token])
    masks = np.array([mask])
    segments = np.array([segment])

    return [tokens, masks, segments]


# #### 음악 생성용 텍스트 여기서 수정

# In[ ]:


text_analysis_result = evaluation_predict()
print("감성 분석 결과:", text_analysis_result)  # 결과 확인

# 감성 분석 결과에서 필요한 감정 단어만 추출하여 사용
text_prompt = text_analysis_result.split("확률 : ")[1].split(" ")[1]  # 감정 단어만 추출
text_prompt = f"{text_prompt} 분위기를 피아노와 오르골 소리로 표현해서 음악 만들어줘."  # 최종 음악 생성 input 문장
print("음악 생성용 입력:", text_prompt)


# ### 음악 파일 생성

# In[ ]:


# pipeline 함수: Hugging Face에서 제공하는 API
from transformers import pipeline
# scipy 라이브러리의 wavfile 모듈을 사용하여 오디오 파일 저장
from scipy.io.wavfile import write  # scipy 라이브러리의 wavfile 모듈을 사용하여 오디오 파일 저장


# In[ ]:


# MusicGen 파이프라인 생성
# "text-to-audio": 파이프라인의 유형(텍스트 입력)
musicgen_pipeline = pipeline("text-to-audio", model="facebook/musicgen-small")


# In[ ]:


# 위 메시지는 모델의 구성정보를 알려주며, MusicGen 모델이 성공적으로 로드되었음을 의미


# In[ ]:


# 입력된 텍스트로 음악 생성
audio_output = musicgen_pipeline(text_prompt)


# In[ ]:


# 위 메시지는 함수가 빈 attention mask 지원하지 않는다는데 실행에는 문제없어서 ㄱㅊ긴함
# 나중에 오류 거슬리면 1. attn_implemention 인자 추가 하거나 2. guidance_scale을 1보다 낮추기


# In[ ]:


# audio 데이터와 샘플링 속도 추출
audio_array = audio_output["audio"]  # 오디오 데이터
sampling_rate = audio_output["sampling_rate"]  # 샘플링 속도


# In[ ]:


# 생성된 오디오 파일을 로컬에 저장
write("generated_music_final.wav", sampling_rate, audio_array)


# In[ ]:


print("음악 생성 완료: generated_music_final.wav 파일이 저장되었습니다.")


# In[ ]:


import os
print(os.getcwd())


# ### 최종 버전 확인

# In[ ]:


get_ipython().system('pip list')

