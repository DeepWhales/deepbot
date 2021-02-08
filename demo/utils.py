import random
import logging

import torch
import numpy as np

from transformers import AutoTokenizer

import pickle

# from transformers import ElectraForTokenClassification, TokenClassificationPipeline
# from utils.tokenization_kocharelectra import KoCharElectraTokenizer

import json
import urllib
from bs4 import BeautifulSoup



#################
## NER 모델 호출 ##
#################
# tokenizer = KoCharElectraTokenizer.from_pretrained("monologg/kocharelectra-base-kmounlp-ner")   # 해양대 NER
# model = ElectraForTokenClassification.from_pretrained("monologg/kocharelectra-base-kmounlp-ner")# 해양대 NER
# ner = TokenClassificationPipeline(
#     model=model, tokenizer=tokenizer, ignore_labels=["O"], grouped_entities=True, device=-1
# )
from pororo import Pororo
ner = Pororo(task="ner", lang="ko")
###

from kocrawl.spell import SpellCrawler


intent_str_dic = {'elevator-on'                     : '엘리베이터 호출',
                'gas-off'                       : '가스 밸브 OFF',
                'gas-on'                        : '가스 밸브 ON',

                'heat-cold-state'                     : '난방 COLD-STATE',
                'heat-down'                     : '난방 DOWN',
                'heat-hot-state'                      : '난방 HOT-STATE',
                'heat-off'                      : '난방 OFF',
                'heat-on'                       : '난방 ON',
                'heat-reservation-cancel'       : '난방 예약 취소',
                'heat-reservation-off-after'    : '난방 OFF 예약',
                'heat-reservation-off-at'       : '난방 OFF 예약',
                'heat-reservation-on-after'     : '난방 ON 예약',
                'heat-reservation-on-at'        : '난방 ON 예약',
                'heat-reservation-state'        : '난방 예약 STATE',
                'heat-state'                    : '난방 STATE',
                'heat-up'                       : '난방 UP',

                'light-on'                      : '조명 ON',
                'light-off'                     : '조명 OFF',

                'parking-location'              : '주차 위치',

                'security-off'                  : '방범 OFF',
                'security-on'                   : '방범 ON',

                'vent-down'                     : '환기 DOWN',
                'vent-off'                      : '환기 OFF',
                'vent-on-high'  : '환기 ON',
                'vent-on-low'   : '환기 ON',
                'vent-on-mid'   : '환기 ON',
                'vent-state'    : '환기 STATE',
                'vent-up'       : '환기 UP',

                'weather'           : '날씨',
                'greeting'          : '인사',
                'search'            : '검색',
              }


def pickle_read(file_name):
    # file_name = 'user.pickle'
    with open(file_name, 'rb') as fr:
        user_loaded = pickle.load(fr)
    return user_loaded

def get_label():
    index2label=pickle_read('index2label.pickle')
    labels = list(index2label.keys())
    return labels

def get_index2label():
    index2label = pickle_read('index2label.pickle')
    return index2label

def get_label2index():
    label2index = pickle_read('label2index.pickle')
    return label2index

def load_tokenizer(model_name_or_path):
    return AutoTokenizer.from_pretrained(model_name_or_path)

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def set_seed(seed, no_cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_score(preds, labels)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_score(preds, labels):
    return {
        "acc": simple_accuracy(preds, labels),
    }

# https://github.com/kmounlp/NER/blob/master/NER%20Guideline%20(ver%201.0).pdf
# 가. 개체명(ENAMEX): 개체명은 주로 고유명사에 해당하며 아래와 같은 형태가 있다.
# 1) PER: 인명(person)
# 2) ORG: 기관/조직(organization)
# 3) LOC: 장소/위치(location)
# 4) POH: 기타 고유명사
# 나.기산표현(TIMEX): 시간표현은 절대적인 시간표현을 기준으로 한다. 절대적인 시간표현이란 '2002년 12월 25일'과 같이 구체적인 시간을 나타내는 표현이며, 상대적인 시간푠혀은 '오늘', '어제', '올해', '금년', '이틀 뒤' 등과 같이 특정 기준이 있어야 정확한 시점을 찾을 수 있는 표현이다.
# 5) DAT: 날자(date)
# 6) TIM: 시간(time)
# 7) DUR: 기간(duration)
# 다. 수량표현(NUMEX): 수량표현은 숫자와 단위를 함계 사용한 표현이나 숫자를 단독으로 사용한 경우를 나타내며 금액, 비율, 기타 숫자표현의 형태로 태깅할 수 있다.
# 1) MNY: 금액(money)
# 2) PNT: 비율(rate)
# 3) NOH: 기타 숫자표현

def preprocess(utterance):

    # 맞춤법 검사기
    spellCheck = SpellCrawler()
    utterance = spellCheck.request(utterance)

    # 집안류
    utterance = utterance.replace('작은방', '집안')
    utterance = utterance.replace('거실', '집안')
    utterance = utterance.replace('안방', '집안')
    utterance = utterance.replace('화장실', '집안')

    # 오탈자 변경
    utterance = utterance.replace('엘레베이터', '엘리베이터')
    utterance = utterance.replace('엘베', '엘리베이터')
    utterance = utterance.replace('스톱', '스탑')
    utterance = utterance.replace('벨브', '밸브')
    utterance = utterance.replace('까스', '가스')

    # 난방류
    utterance = utterance.replace('보일러', '난방')


    # 글자 묶어 주기
    utterance_ner = ner(utterance)

    # {'PS': 'PERSON', 'LC': 'LOCATION', 'OG': 'ORGANIZATION', 'AF': 'ARTIFACT', 'DT': 'DATE', 'TI': 'TIME',
    #  'CV': 'CIVILIZATION', 'AM': 'ANIMAL', 'PT': 'PLANT', 'QT': 'QUANTITY', 'FD': 'STUDY_FIELD', 'TR': 'THEORY',
    #  'EV': 'EVENT', 'MT': 'MATERIAL', 'TM': 'TERM'}

    if len(utterance_ner) != 0:
        utterance_new = ''
        for word_ner in utterance_ner:
            if 'PERSON' == word_ner[1]:
                utterance_new = utterance_new + '사람이름'
            elif 'TIME' == word_ner[1]:
                utterance_new = utterance_new + '시간'
            elif 'DATE' == word_ner[1]:
                utterance_new = utterance_new + '날짜'
            else:
                utterance_new = utterance_new + word_ner[0]

        utterance = utterance_new


    else:
        pass



    # 숫자류
    utterance = utterance.replace('1', '0')
    utterance = utterance.replace('2', '0')
    utterance = utterance.replace('3', '0')
    utterance = utterance.replace('4', '0')
    utterance = utterance.replace('5', '0')
    utterance = utterance.replace('6', '0')
    utterance = utterance.replace('7', '0')
    utterance = utterance.replace('8', '0')
    utterance = utterance.replace('9', '0')


    # 자동차류
    utterance = utterance.replace('쏘나타', '자동차')
    utterance = utterance.replace('아벤떼', '자동차')
    utterance = utterance.replace('그렌져', '자동차')

    return utterance


if __name__ == '__main__':
    print(preprocess('우리집 어때?'))
    print(preprocess('내일 오전에 보일러 켜줘'))
    print(preprocess('내일 오전 10시 10분에 보일러 켜줘'))
    print(preprocess('수요일에 보일러 켜줘'))
    print(preprocess('문제인이 누구야?'))
    print(preprocess('방법 켜줘'))
