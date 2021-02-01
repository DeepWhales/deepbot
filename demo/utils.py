import os
import random
import logging

import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel, AutoConfig

import pickle


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

def preprocess(utterance):
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

    # 난방류
    utterance = utterance.replace('보일러', '난방')

    # 자동차류
    utterance = utterance.replace('쏘나타', '자동차')

    # 집안류
    # utterance = utterance.replace('거실', '집안')
    # utterance = utterance.replace('안방', '집안')
    # utterance = utterance.replace('화장실', '집안')

    
    # 사람 이름류

    
    # 오탈자 변경
    utterance = utterance.replace('엘레베이터', '엘리베이터')
    utterance = utterance.replace('스톱', '스탑')
    utterance = utterance.replace('벨브', '밸브')
    utterance = utterance.replace('까스', '가스')
    





    return utterance