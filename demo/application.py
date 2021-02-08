import os
import sys

sys.path.append('..')

from flask import Flask
from flask import render_template

# from kochat.data import Dataset
from kochat.app.scenario_manager import ScenarioManager
# from kochat.loss import CRFLoss, CenterLoss
# from kochat.model import intent, embed, entity
# from kochat.proc import DistanceClassifier, GensimEmbedder, EntityRecognizer

from demo.scenrios import restaurant, travel, dust, weather
# from scenrios import restaurant, travel, dust, weather

import pickle
import torch

from transformers import AutoTokenizer
from transformers import BertForSequenceClassification

# import pandas as pd
#
# import time
import threading

from datetime import datetime

from utils import intent_str_dic, preprocess

###############
# 피드백 메모리 #
###############
# feedback = pd.DataFrame(columns=['utterance, intent, label'])
feedback = {'id':[], 'text':[], 'utterance':[], 'intent':[], 'label':[], 'score':[]}
id_global = ''
text_global = ''
utterence_global = ''
intent_global = ''
label_global = ''
score_global = ''


def feedback_write():
    # today1 = datetime.today().strftime("%Y%m%d%H%M%S")
    today = datetime.today().strftime("%Y_%m%d_%H%M")

    # https: // tre2man.tistory.com / 188
    print( today + ' feedback write')
    # df_feedback = pd.DataFrame(feedback)
    # df_feedback.to_pickle('saved/' + today + '.pkl')

    # save data
    with open('saved/' + today + '.pkl', 'wb') as fw:
        pickle.dump(feedback, fw)

    # # load data
    # with open('saved/' + today + '.pkl', 'rb') as fr:
    #     data_loaded = pickle.load(fr)

    feedback['id'].clear()
    feedback['text'].clear()
    feedback['utterance'].clear()
    feedback['intent'].clear()
    feedback['label'].clear()
    feedback['score'].clear()
    # threading을 정의한다. 5ㅊ오마다 반복 수행함
    # threading.Timer(360, feedback_write).start() # 초단위 저장 (한시간 마다 저장)
    threading.Timer(720, feedback_write).start() # 초단위 저장 (두시간 마다 저장)
    # threading.Timer(61, feedback_write).start()  # 초단위 저장 (한시간 마다 저장)


########################
# intent to tntent_str #
########################
# intent_str_dic = {'elevator-on'                     : '엘리베이터 호출',
#                 'gas-off'                       : '가스 밸브 OFF',
#                 'gas-on'                        : '가스 밸브 ON',
#
#                 'heat-cold-state'                     : '난방 COLD-STATE',
#                 'heat-down'                     : '난방 DOWN',
#                 'heat-hot-state'                      : '난방 HOT-STATE',
#                 'heat-off'                      : '난방 OFF',
#                 'heat-on'                       : '난방 ON',
#                 'heat-reservation-cancel'       : '난방 예약 취소',
#                 'heat-reservation-off-after'    : '난방 OFF 예약',
#                 'heat-reservation-off-at'       : '난방 OFF 예약',
#                 'heat-reservation-on-after'     : '난방 ON 예약',
#                 'heat-reservation-on-at'        : '난방 ON 예약',
#                 'heat-reservation-state'        : '난방 예약 STATE',
#                 'heat-state'                    : '난방 STATE',
#                 'heat-up'                       : '난방 UP',
#
#                 'light-on'                      : '조명 ON',
#                 'light-off'                     : '조명 OFF',
#
#                 'parking-location'              : '주차 위치',
#
#                 'security-off'                  : '방범 OFF',
#                 'security-on'                   : '방범 ON',
#
#                 'vent-down'                     : '환기 DOWN',
#                 'vent-off'                      : '환기 OFF',
#                 'vent-on-high'  : '환기 ON',
#                 'vent-on-low'   : '환기 ON',
#                 'vent-on-mid'   : '환기 ON',
#                 'vent-state'    : '환기 STATE',
#                 'vent-up'       : '환기 UP',
#
#                 'weather'           : '날씨',
#                 'greeting'          : '인사',
#                 'search'            : '검색',
#               }


####################################
## 라이브러리 한방 설치
## pip install -r requirements.txt #
####################################

model_name_or_path = 'beomi/kcbert-base'
model_dir = 'model'
device = 'cpu'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model_class = BertForSequenceClassification

# with open('indx2label.pkl', 'rb') as fr:
#     indx2label = pickle.load(fr)

####################
## 2. 모델 불러오기 ##
####################
model = model_class.from_pretrained(model_dir)
model.to(device)

indx2label = model.config.id2label


# 데이터 관리
# dataset = Dataset()

# 임베딩
# embed_processor = GensimEmbedder(model=embed.FastText())
# embed_processor = GensimEmbedder(model=embed.Word2Vec())

# 인텐트 분류
# intent_classifier = DistanceClassifier(
#     model=intent.CNN(dataset.intent_dict),
#     loss=CenterLoss(dataset.intent_dict),
# )

# 개체명 인식
# entity_recognizer = EntityRecognizer(
#     model=entity.LSTM(dataset.entity_dict),
#     loss=CRFLoss(dataset.entity_dict)
# )

# 시나리오
scenario_manager = ScenarioManager([
    weather, dust, travel, restaurant
])

# 학습 여부
# train = False
# if train:
#
#     embed_processor.fit(dataset.load_embed())
#     # intent_data = dataset.load_intent(embed_processor)
#     # entity_data = dataset.load_entity(embed_processor)
#     intent_classifier.fit(dataset.load_intent(embed_processor))
#     # entity_recognizer.fit(dataset.load_entity(embed_processor))

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
dialogue_cache = {}

@app.route('/')
def index():
    return render_template("index.html")

# http://192.168.10.102:8080/request_chat/이름/내용
@app.route('/request_chat/<uid>/<text>', methods=['GET'])
def request_chat(uid: str, text: str) -> dict:
    # print(uid)
    # print(text)
    # prep = dataset.load_predict(text, embed_processor)
    # print(prep)
    # intent = intent_classifier.predict(prep, calibrate=False)
    # entity = entity_recognizer.predict(prep)
    # entity = None
    # text = dataset.prep.tokenize(text, train=False)
    # dialogue_cache[uid] = scenario_manager.apply_scenario(intent, entity, text)

    # BERT 인텐트
    # utterance =
    utterence = preprocess(text)
    max_seq_len = 50
    inputs = tokenizer.encode_plus(utterence,
            None,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=max_seq_len,
    )

    ids = inputs["input_ids"]
    token_type_ids = inputs["token_type_ids"]
    mask = inputs["attention_mask"]

    input_data = {
        'ids': torch.tensor(ids, dtype=torch.long),
        'mask': torch.tensor(mask, dtype=torch.long),
        'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        # 'target': torch.tensor(self.train_csv.iloc[index, 2], dtype=torch.long)
        # 'target': torch.tensor(self.target[index], dtype=torch.long)
    }

    input_data['ids'] = input_data['ids'].to(device)
    input_data['mask'] = input_data['mask'].to(device)
    input_data['token_type_ids'] = input_data['token_type_ids'].to(device)
    # input_data['target'] = input_data['target'].to(device)

    input_data['ids'] = input_data['ids'].unsqueeze(0)
    input_data['mask'] = input_data['mask'].unsqueeze(0)
    input_data['token_type_ids'] = input_data['token_type_ids'].unsqueeze(0)

    # 3. 모델에 데이터 넣기
    inputs = {'input_ids': input_data['ids'],
              'token_type_ids' : input_data['token_type_ids'],
              'attention_mask': input_data['mask']
              }

    outputs = model(**inputs)

    inference_value = torch.argmax(outputs.logits)
    intent = indx2label[inference_value.item()]

    entity = None
    # dialogue_cache[uid] = scenario_manager.apply_scenario(intent, entity, text)
    # return dialogue_cache[uid]

    # print(text)
    # print(dialogue_cache[uid]['intent'])

    output_topk5 = outputs.logits.topk(3)

    intent1 = indx2label[output_topk5.indices[0][0].item()]
    # intent2 = indx2label[output_topk5.indices[0][1].item()]
    # intent3 = indx2label[output_topk5.indices[0][2].item()]
    # intent = intent1 + ', ' + intent2 + ', ' + intent3
    intent = intent1

    total_sum = output_topk5.values[0][:].sum()

    score1 = output_topk5.values[0][0].item()/total_sum
    # score2 = output_topk5.values[0][1].item()/total_sum
    # score3 = output_topk5.values[0][2].item()/total_sum
    score1_str = str(score1.item())[0:6]
    # score2_str = str(score2.item())[0:6]
    # score3_str = str(score3.item())[0:6]
    # score_str = score1_str + ', ' + score2_str + ', ' + score3_str
    score_str = score1_str

    # feedback = {: [], 'intent': [], 'label': [], 'score': []};
    id_global = uid
    utterence_global = utterence
    intent_global = intent
    label_global = ''
    score_global = outputs.logits.squeeze()
    # feedback['id'] = id_global
    # feedback['utterance'] = utterence_global
    # feedback['intent'] = intent_global
    # feedback['score'] = score_global
    feedback['id'].append(id_global)
    feedback['text'].append(text_global)
    feedback['utterance'].append(utterence_global)
    feedback['intent'].append(intent_global)
    feedback['score'].append(score_global)
    feedback['label'].append(999)

    intent_str =  intent_str_dic[intent]

    if score1.item() > 0.7:
        intent_str = intent_str
    elif score1.item() >= 0.55 and score1.item() <= 0.7:
        intent_str = '질문하신 의도가 (' + intent_str + ')이 맞나요?'
    else:
        intent_str = intent_str + '스마트홈에 관련된 질문을 부탁드립니다. (난방, 주차위치, 가스 밸브, 조명, 방범, 환기, 날씨, 간단한 인사, 검색)'

    dialogue_cache = {'input': text, 'intent': intent_str, 'entity': entity, 'state':'FALLBACK', 'answer': None, 'score': score_str}

    # feedback에 저장
    # feedback['']
    return dialogue_cache

# http://192.168.10.102:8080/request_chat/이름/내용
@app.route('/request_correct/<uid>/', methods=['GET'])
def request_correct(uid: str) -> dict:
    # print('python 함수 통과')

    feedback_len = len(feedback['label'])
    feedback['label'][feedback_len - 1] = 1

    text = '평가해주셔서 감사합니다.'
    dialogue_cache = {'input': text}
    return dialogue_cache

@app.route('/request_incorrect/<uid>/', methods=['GET'])
def request_incorrect(uid: str) -> dict:
    # print('python 함수 통과')

    feedback_len = len(feedback['label'])
    feedback['label'][feedback_len - 1] = 0

    text = '평가해주셔서 감사합니다.'
    dialogue_cache = {'input': text}
    return dialogue_cache

# @app.route('/fill_slot/<uid>/<text>', methods=['GET'])
# def fill_slot(uid: str, text: str) -> dict:
#     print('fill_slot')
#     prep = dataset.load_predict(text, embed_processor)
#     entity = entity_recognizer.predict(prep)
#     text = dataset.prep.tokenize(text, train=False)
#     intent = dialogue_cache[uid]['intent']
#
#     text = text + dialogue_cache[uid]['input']  # 이전에 저장된 dict 앞에 추가
#     entity = entity + dialogue_cache[uid]['entity']  # 이전에 저장된 dict 앞에 추가
#
#     answer_1 = scenario_manager.apply_scenario(intent, entity, text)
#     return answer_1

# .js파일 익스플로에서 코드 적용안될 때 (컨트롤 +쉬프트 + r)
if __name__ == '__main__':
    feedback_write()
    app.run(port=8080, host='127.0.0.1', debug=True)
    # app.run(port=8080, host='0.0.0.0')
    # app.run(port=5605, host='192.168.0.64')