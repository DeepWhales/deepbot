import torch

from transformers import AutoTokenizer, BertForSequenceClassification

import pandas as pd
import pickle
import openpyxl

from utils.utils import preprocess, postprocess

import torch.nn as nn

if __name__ == '__main__':

    # df_data = pd.read_excel(
    #     'D:/work/project/chatbot_AI_hub/reference/KcBERT-nsmc_yoon/data/pvot/pvot_dataset_train.xlsx')
    # # idx2label
    # df_data1 = df_data.loc[:, ['target_2', 'target']]
    # df_data2 = df_data1.drop_duplicates()
    # indx2label = {}
    #
    # for index, row in df_data2.iterrows():
    #     # indx2label[row['target_2']] = row['target']
    #     indx2label[row['target']] = row['target_2']

    # 0. 테스트할 데이터 불러 오기
    # df_data = pd.read_excel('./feedback_분석/feedback.xlsx', engine="openpyxl")
    df_data = pd.read_excel('./feedback_분석/test_dataset.xlsx', engine="openpyxl")

    # 1. index를 label로 변환 시켜줄 dictionary 불러 오기
    # with open('indx2label.pkl', 'rb') as fr:
    #     indx2label = pickle.load(fr)

    # 2. model ,tokenizer 설정 및 weight 불러 오기
    model_name_or_path = 'beomi/kcbert-base'
    # model_dir = 'model/2021_02_04_17_44'
    model_dir = 'model/2021_02_10_12_27'
    device = 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model_class = BertForSequenceClassification

    model = model_class.from_pretrained(model_dir)
    model.to(device)

    indx2label = model.config.id2label

    # 3. model 테스트 및 결과 보기
    result = {'id':[], 'utterance': [], 'output': [], 'output한글':[], 'output_score': []}

    for index, rows in df_data.iterrows():
        id = rows['id']
        utterance = rows['utterance']
        utterance = preprocess(utterance)

        # utterance = '안녕하세요'
        # utterance = '병원 가니?'
        # utterance = '벨브 그만'
        # utterance = '날씨가 어때요?'

        max_seq_len = 50
        special_tokens_count = 2
        sequence_a_segment_id = 0
        cls_token_segment_id = 0
        mask_padding_with_zero = True
        pad_token_segment_id = 0

        # Setting based on the current model type
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        pad_token_id = tokenizer.pad_token_id

        tokens = tokenizer.tokenize(utterance)

        # Account for [CLS] and [SEP]

        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        inputs1 = 0

        ###########################################

        inputs = tokenizer.encode_plus(utterance,
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
        ####################
        ## 2. 모델 불러오기 ##
        ####################


        # 3. 모델에 데이터 넣기
        inputs = {'input_ids': input_data['ids'],
                  'token_type_ids': input_data['token_type_ids'],
                  'attention_mask': input_data['mask']
                  }

        outputs = model(**inputs)


        intent_str, intent_candidate_str, score_str = postprocess(outputs, model)

        print('인텐트:', intent_str)
        print(intent_candidate_str)
        print(score_str)



        m = nn.Softmax(dim=1)
        output_softmax = m(outputs.logits)



        if False:   # 그래프 on, off


            print(output_softmax.topk(3).values.tolist())
            output_softmax_value = output_softmax.squeeze()
            xx = range(0, 31)
            plt.plot(xx, output_softmax_value.tolist())
            plt.show()

        aa = 0
        result['id'].append(id)
        result['utterance'].append(utterance)
        result['output'].append(intent)
        result['output한글'].append(intent_str_dic[intent1])
        result['output_score'].append(output_softmax.max().item())
        aaa = 0

    df_result = pd.DataFrame(result)

    # df_result.to_excel('prediction_result.xlsx', engine="openpyxl")
    df_result.to_excel('prediction_result.xlsx')

    # print(index2label[inference_value.item()])
    #
    # # b = pickle_read('label2index.pickle')
    # aaaa =0
    #
    # topK = 10
    # item_score_dict = {'일':3, '이':2, '삼':5, '사':1, '오':3}
    #
    # ranklist = heapq.nlargest(topK, item_score_dict, key=item_score_dict.get)

    # print(ranklist)
    #
    #
    # print(ranklist[0], item_score_dict[ranklist[0]])