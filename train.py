
import random
import argparse

import numpy as np
import torch

from torch.utils.data import (
    DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from pytorch_transformers import *

import data


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default='woz/woz_train_en.json', type=str, required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default='.', type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_len", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_steps", default=100, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=10, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    return parser.parse_args()


def load_data(args):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    examples = list(data.InputExample.read_woz(args.data_path, 'train'))
    features = [data.InputFeature.make(example, tokenizer, args.max_len) for example in examples]

    all_input_ids   = torch.tensor([f.input_ids   for f in features], dtype=torch.long)
    all_input_mask  = torch.tensor([f.input_mask  for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_start_pos   = torch.tensor([f.start_pos   for f in features], dtype=torch.long)
    all_end_pos     = torch.tensor([f.end_pos     for f in features], dtype=torch.long)
    all_class_label_id = torch.tensor([f.class_label_id    for f in features], dtype=torch.long)

    return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_pos, all_end_pos, class_label_id)




def main():
    args = arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    
    model = BertModel.from_pretrained('bert-base-uncased')


    model.to(args.device)
    global_step, tr_loss = train(args, train_dataset, model, tokenizer)



if __name__ == "__main__":
    main()

