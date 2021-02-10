from transformers import AutoTokenizer, BertForSequenceClassification
from utils.utils import preprocess, postprocess

import torch

if __name__ == '__main__':
    # model ,tokenizer 설정 및 weight 불러 오기
    model_name_or_path = 'beomi/kcbert-base'
    # model_dir = 'model/2021_02_04_17_44'
    model_dir = 'model/2021_02_10_12_27'
    device = 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model_class = BertForSequenceClassification

    model = model_class.from_pretrained(model_dir)
    model.to(device)
    model.eval()


    for step in range(15):

        utterance = input(">> User:")
        utterance_pre = preprocess(utterance)
        print(utterance_pre)
        tokens = tokenizer.tokenize(utterance_pre)
        print(tokens)
        # tokens = tokenizer.encode(utterance_pre)
        # print(tokens)

        inputs = tokenizer.encode_plus(utterance_pre,
                                       None,
                                       pad_to_max_length=True,
                                       add_special_tokens=True,
                                       return_attention_mask=True,
                                       max_length=50,
                                       )
        ids             = inputs["input_ids"]
        token_type_ids  = inputs["token_type_ids"]
        mask            = inputs["attention_mask"]

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
        inputs = {'input_ids': input_data['ids'],
                  'token_type_ids': input_data['token_type_ids'],
                  'attention_mask': input_data['mask']
                  }

        outputs = model(**inputs)
        # print(outputs)

        intent_str, intent_candidate_str, score_str = postprocess(outputs, model)

        print('인텐트:', intent_str)
        print(intent_candidate_str)
        print(score_str)
        aaaa = 0
