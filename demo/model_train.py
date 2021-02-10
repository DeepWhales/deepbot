from utils.utils import init_logger, set_seed
# from utils import init_logger, load_tokenizer, set_seed
# from data_loader import load_and_cache_examples

import logging
from tqdm import tqdm, trange

import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoConfig, BertForSequenceClassification
from transformers import AutoTokenizer

# from utils import compute_metrics, get_label, get_index2label, get_label2index

from utils.utils import compute_metrics, preprocess, get_label

from datetime import datetime

logger = logging.getLogger(__name__)

import os
import copy
import json

import torch
from torch.utils.data import TensorDataset

import pandas as pd
import openpyxl
# import xlrd

class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, label):
        self.guid = guid
        self.text_a = text_a
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class PvotProcessor(object):
    """Processor for the NSMC data set """

    def __init__(self, train_file, dev_file, test_file, data_dir):
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.data_dir = data_dir

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        df_lines = pd.read_excel(input_file, engine='openpyxl')
        return df_lines
        # with open(input_file, "r", encoding="utf-8") as f:
        #     lines = []
        #     for line in f:
        #         lines.append(line.strip())
        #     return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        # examples = []
        examples = lines
        # for (i, line) in enumerate(lines[1:]):
        #     line = line.split('\t')
        #     guid = "%s-%s" % (set_type, i)
        #     text_a = line[1]
        #     label = int(line[2])
        #     if i % 1000 == 0:
        #         logger.info(line)
        #     examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.train_file
        elif mode == 'dev':
            file_to_read = self.dev_file
        elif mode == 'test':
            file_to_read = self.test_file

        logger.info("LOOKING AT {}".format(os.path.join(self.data_dir, file_to_read)))
        text_data_path = os.path.join(self.data_dir, file_to_read)
        text_read_data = self._read_file(text_data_path)
        data_tmp = self._create_examples(text_read_data, mode)
        return data_tmp

processors = {
    "pvot": PvotProcessor,
}

def convert_examples_to_features(df_examples, max_seq_len, tokenizer,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    # for example in df_examples.iterrows():
    for ex_index, (label, utterance) in enumerate(zip(df_examples['target'], df_examples['utterance'])):

        utterance = preprocess(utterance)
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(df_examples)))

        tokens = tokenizer.tokenize(utterance)

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
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

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

        label_id = label

        if ex_index < 5:
            logger.info("*** Example ***")
            # logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label_id=int(label_id)
                          ))

    return features

def load_and_cache_examples(task, tokenizer, mode, model_name_or_path, max_seq_len, data_dir, train_file, dev_file, test_file):
    processor = processors[task](train_file, dev_file, test_file, data_dir)

    # Load data features from cache or dataset file
    cached_file_name = 'cached_{}_{}_{}_{}'.format(
        task, list(filter(None, model_name_or_path.split("/"))).pop(), max_seq_len, mode)

    cached_features_file = os.path.join(data_dir, cached_file_name)
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        features = convert_examples_to_features(examples, max_seq_len, tokenizer)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_label_ids)
    return dataset

class Trainer(object):
    def __init__(self, model_name_or_path, task, no_cuda, max_steps, train_batch_size, gradient_accumulation_steps, weight_decay, learning_rate,
                 adam_epsilon, warmup_steps, max_grad_norm, logging_steps, save_steps, eval_batch_size, model_dir,
                 train_dataset=None, dev_dataset=None, test_dataset=None,
                 data_path=None):
        # self.args = args
        self.max_steps = max_steps
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.train_batch_size = train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps

        self.adam_epsilon = adam_epsilon
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_batch_size = eval_batch_size

        self.model_dir = model_dir

        self.label_lst = get_label(data_path)
        self.num_labels = len(self.label_lst)

        self.config_class = AutoConfig
        self.model_class = BertForSequenceClassification

        # in2label_1={str(i): label for i, label in enumerate(self.label_lst)}
        # label2id_1 = {label: i for i, label in enumerate(self.label_lst)}
        # self.config = self.config_class.from_pretrained(model_name_or_path,
        #                                                 num_labels=self.num_labels,
        #                                                 finetuning_task=task,
        #                                                 id2label={str(i): label for i, label in enumerate(self.label_lst)},
        #                                                 label2id={label: i for i, label in enumerate(self.label_lst)})


        # self.config = self.config_class.from_pretrained(model_name_or_path,
        #                                                 num_labels=self.num_labels,
        #                                                 finetuning_task=task,
        #                                                 id2label=index2label,
        #                                                 label2id=label2index)
        # self.config = self.config_class.from_pretrained(model_name_or_path,
        #                                                 num_labels=self.num_labels,
        #                                                 finetuning_task=task)
        self.config = self.config_class.from_pretrained(model_name_or_path,
                                                        num_labels=self.num_labels,
                                                        finetuning_task=task,
                                                        id2label={str(i): label for i, label in enumerate(self.label_lst)},
                                                        label2id={label: i for i, label in enumerate(self.label_lst)})

        self.model = self.model_class.from_pretrained(model_name_or_path, config=self.config)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.train_batch_size)

        if self.max_steps > 0:
            t_total = self.max_steps
            self.num_train_epochs = self.max_steps // (len(train_dataloader) // self.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.gradient_accumulation_steps * self.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.num_train_epochs)
        logger.info("  Total train batch size = %d", self.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.logging_steps)
        logger.info("  Save steps = %d", self.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                loss = outputs[0]
                # self.save_model()
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()

                if (step + 1) % self.gradient_accumulation_steps == 0:  # 0으로 떨어 질 때마다 업데이트
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.logging_steps > 0 and global_step % self.logging_steps == 0:
                        self.evaluate("test")  # Only test set available for NSMC

                    if self.save_steps > 0 and global_step % self.save_steps == 0:
                        self.save_model()
                        # save_model()

                if 0 < self.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                inputs['token_type_ids'] = batch[2]

                outputs = self.model(**inputs)

                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        preds = np.argmax(preds, axis=1)

        preds_reshape = preds.tolist()
        out_label_ids_reshape = out_label_ids.tolist()
        class_labels = list(self.model.config.label2id.keys()) # classification 문제에만 들어가는 인자
        result = compute_metrics(preds_reshape, out_label_ids_reshape, class_labels)
        print(result)
        # results.update(result)

        # # Slot result
        # # preds2 = np.argmax(preds, axis=2)
        # slot_label_map = {i: label for i, label in enumerate(self.label_lst)}
        # out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        # preds_list = [[] for _ in range(out_label_ids.shape[0])]
        #
        # for i in range(out_label_ids.shape[0]):
        #     for j in range(out_label_ids.shape[1]):
        #         if out_label_ids[i, j] != self.pad_token_label_id:
        #             out_label_list[i].append(slot_label_map[out_label_ids[i][j]])
        #             preds_list[i].append(slot_label_map[preds[i][j]])
        #
        # result = compute_metrics(out_label_list, preds_list)
        # results.update(result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results

    def save_model(self):
        # time_str = datetime.today().strftime("%Y%m%d%H%M%S")
        time_str = datetime.today().strftime("%Y_%m_%d_%H_%M")
        folder_name = self.model_dir + '/' + time_str
        # Save model checkpoint (Overwrite)
        # if not os.path.exists(self.model_dir):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        self.model_dir = folder_name
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.model_dir)

    def load_model(self):
        # Check whether model exists
        print(self.model_dir)
        if not os.path.exists(self.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(self.model_dir)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")

if __name__ == '__main__':

    # pd_data = pd.read_excel('./data/pvot/pvot_dataset_target.xlsx')

    # task = 'nsmc'                       # parser.add_argument("--task", default="nsmc", type=str, help="The name of the task to train")
    task = 'pvot'                         # parser.add_argument("--task", default="nsmc", type=str, help="The name of the task to train")
    model_dir = "./model"                   # parser.add_argument("--model_dir", default="./model", type=str, help="Path to save, load model")

    # data_dir = "./data/nsmc"            # parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    data_dir = "./data/pvot"              # parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")

    # train_file = "pvot_dataset_train.xlsx"                    # parser.add_argument("--train_file", default="ratings_train.txt", type=str, help="Train file")
    # train_file = "pvot_dataset_target.xlsx"                   # parser.add_argument("--train_file", default="ratings_train.txt", type=str, help="Train file")
    # test_file = "pvot_dataset_test.xlsx"                      # parser.add_argument("--test_file", default="ratings_test.txt", type=str, help="Test file")
    train_file = "pvot_dataset_target_high_베이스.xlsx"          # parser.add_argument("--train_file", default="ratings_train.txt", type=str, help="Train file")
    # train_file = "pvot_dataset_target_high_베이스_small.xlsx"      # parser.add_argument("--train_file", default="ratings_train.txt", type=str, help="Train file")
    test_file = "pvot_dataset_target_high_test.xlsx"                # parser.add_argument("--test_file", default="ratings_test.txt", type=str, help="Test file")

    data_path = os.path.join(data_dir, train_file)
    # idx2label = make_idx2label(data_path)

    dev_file = ".txt"

    model_name_or_path = 'beomi/kcbert-base'    # parser.add_argument("--model_name_or_path", default="kobert", type=str)
    seed = 42                                   # parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    train_batch_size = 8                        # parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    eval_batch_size = 8                         # parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    max_seq_len = 50                            # parser.add_argument("--max_seq_len", default=50, type=int, help="The maximum total input sequence length after tokenization.")
    learning_rate = 5e-5                        # parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    num_train_epochs = 5.0                      # parser.add_argument("--num_train_epochs", default=5.0, type=float, help="Total number of training epochs to perform.")
    weight_decay = 0.0                          # parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    gradient_accumulation_steps = 1             # parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")

    adam_epsilon = 1e-8     # parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    max_grad_norm = 1.0     # parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # max_steps = 1         # parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    max_steps = 2001        # parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")

    warmup_steps = 0    # parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    logging_steps = 2000        # parser.add_argument('--logging_steps', type=int, default=2000, help="Log every X updates steps.")
    save_steps = 2000           # parser.add_argument('--save_steps', type=int, default=2000, help="Save checkpoint every X updates steps.")

    do_train = True     # parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    do_eval = False      # parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    no_cuda = False     # parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    # args = parser.parse_args()
    # args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    # main(args)
    init_logger()
    set_seed(seed, no_cuda)

    # 토큰나이져 설정
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # tokenizer = load_tokenizer(model_name_or_path)

    # 데이터 불러 오기
    mode = "train"
    train_dataset = load_and_cache_examples(task, tokenizer, mode, model_name_or_path, max_seq_len, data_dir, train_file, dev_file, test_file)
    dev_dataset = None
    mode = "test"
    test_dataset = load_and_cache_examples(task, tokenizer, mode, model_name_or_path, max_seq_len, data_dir, train_file, dev_file, test_file)

    # 학습 running
    trainer = Trainer(model_name_or_path, task, no_cuda, max_steps, train_batch_size, gradient_accumulation_steps, weight_decay, learning_rate,
                 adam_epsilon, warmup_steps, max_grad_norm, logging_steps, save_steps, eval_batch_size, model_dir, train_dataset, dev_dataset, test_dataset,
                data_path)

    if do_train:
        global_step, avr_loss = trainer.train()

    if do_eval:
        trainer.load_model()
        trainer.evaluate("test")