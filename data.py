
import random
import json

import numpy as np

import tokenization

import utils


SEMANTIC_DICT = {
    'center': ['centre', 'downtown', 'central', 'down town', 'middle'],
    'south': ['southern', 'southside'],
    'north': ['northern', 'uptown', 'northside'],
    'west': ['western', 'westside'],
    'east': ['eastern', 'eastside'],
    'east side': ['eastern', 'eastside'],

    'cheap': ['low price', 'inexpensive', 'cheaper', 'low priced', 'affordable',
                'nothing too expensive', 'without costing a fortune', 'cheapest',
                'good deals', 'low prices', 'afford', 'on a budget', 'fair prices',
                'less expensive', 'cheapeast', 'not cost an arm and a leg'],
    'moderate': ['moderately', 'medium priced', 'medium price', 'fair price',
                'fair prices', 'reasonable', 'reasonably priced', 'mid price',
                'fairly priced', 'not outrageous','not too expensive',
                'on a budget', 'mid range', 'reasonable priced', 'less expensive',
                'not too pricey', 'nothing too expensive', 'nothing cheap',
                'not overpriced', 'medium', 'inexpensive'],
    'expensive': ['high priced', 'high end', 'high class', 'high quality',
                    'fancy', 'upscale', 'nice', 'fine dining', 'expensively priced'],

    'afghan': ['afghanistan'],
    'african': ['africa'],
    'asian oriental': ['asian', 'oriental'],
    'australasian': ['australian asian', 'austral asian'],
    'australian': ['aussie'],
    'barbeque': ['barbecue', 'bbq'],
    'basque': ['bask'],
    'belgian': ['belgium'],
    'british': ['cotto'],
    'canapes': ['canopy', 'canape', 'canap'],
    'catalan': ['catalonian'],
    'corsican': ['corsica'],
    'crossover': ['cross over', 'over'],
    'gastropub': ['gastro pub', 'gastro', 'gastropubs'],
    'hungarian': ['goulash'],
    'indian': ['india', 'indians', 'nirala'],
    'international': ['all types of food'],
    'italian': ['prezzo'],
    'jamaican': ['jamaica'],
    'japanese': ['sushi', 'beni hana'],
    'korean': ['korea'],
    'lebanese': ['lebanse'],
    'north american': ['american', 'hamburger'],
    'portuguese': ['portugese'],
    'seafood': ['sea food', 'shellfish', 'fish'],
    'singaporean': ['singapore'],
    'steakhouse': ['steak house', 'steak'],
    'thai': ['thailand', 'bangkok'],
    'traditional': ['old fashioned', 'plain'],
    'turkish': ['turkey'],
    'unusual': ['unique and strange'],
    'venetian': ['vanessa'],
    'vietnamese': ['vietnam', 'thanh binh'],
}

class_types = ['none', 'dontcare', 'copy_value', 'unpointable']


def fix_label(label):
    fix_table = {'centre': 'center', 'areas': 'area', 'phone number': 'number'}
    return fix_table.get(label.strip(), label.strip())


def find_label_pos(label, sys_utt_tokens, usr_utt_tokens):
    for _label in [label] + SEMANTIC_DICT.get(label, []):
        _label_tokens = utils.tokenize(_label)
        sys_label_pos = list(utils.find_sublist(sys_utt_tokens, _label_tokens))
        usr_label_pos = list(utils.find_sublist(usr_utt_tokens, _label_tokens))
        
        if sys_label_pos or usr_label_pos:
            return _label, sys_label_pos, usr_label_pos

    return label, sys_label_pos, usr_label_pos


def get_utt_label(label, sys_utt_tokens, usr_utt_tokens):
    sys_utt_label = [0 for _ in sys_utt_tokens]
    usr_utt_label = [0 for _ in usr_utt_tokens]

    if label in ('none', 'dontcare'):
        return sys_utt_label, usr_utt_label, label

    label, sys_label_pos, usr_label_pos = \
        find_label_pos(label, sys_utt_tokens, usr_utt_tokens)

    # only take the last occurrence of label in a utterance
    if usr_label_pos:
        s, e = usr_label_pos[-1]
        usr_utt_label[s:e] = [1]*(e-s)
        return sys_utt_label, usr_utt_label, 'copy_value'
    elif sys_label_pos:
        s, e = sys_label_pos[-1]
        sys_utt_label[s:e] = [1]*(e-s)
        return sys_utt_label, usr_utt_label, 'copy_value'
    else:
        return sys_utt_label, usr_utt_label, 'unpointable'
    

class InputExample(object):
    def __init__(self, guid, utt, label_dict={}):
        self.guid = guid
        self.utt = utt
        self.label_dict = label_dict
    
    def __str__(self):
        return "\nguid: " + str(self.guid) \
            + "\nutt: " + str(self.utt) \
            + "\nlabel: " + str(self.label_dict)

    @classmethod
    def read_woz(cls, filepath, split='train', exclude_unpointable=True):
        slot_list = ['area', 'food', 'price range'] # for woz
        for dialog in json.load(open(filepath)):
            for turn in dialog['dialogue']:
                guid = split+'-'+str(dialog['dialogue_idx'])+'-'+str(turn['turn_idx'])
                turn_label = {fix_label(s): fix_label(v) for s, v in turn['turn_label']}
                sys_utt_tokens = utils.tokenize(turn['system_transcript'])
                usr_utt_tokens = utils.tokenize(turn['transcript'])
                label_dict = {slot: get_utt_label(turn_label.get(slot, 'none'), 
                                                  sys_utt_tokens, 
                                                  usr_utt_tokens) for slot in slot_list}

                example = cls(guid, (sys_utt_tokens, usr_utt_tokens), label_dict)
                if [v for v in label_dict.values() if v[0] == 'unpointable']:
                    print('\n<Unpointable>'+str(example))
                    if not exclude_unpointable:
                        yield example
                else:
                    yield example


class InputFeatures(object):
    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_pos,
                 end_pos,
                 class_label_id,
                 is_real_example=True,
                 guid="NONE"):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.class_label_id = class_label_id
        self.is_real_example = is_real_example

    @classmethod
    def make(cls, example, tokenizer, max_len):
        sys_utt_tokens, usr_utt_tokens = example.utt
        joint_sys_utt_label = np.array([x for x, _, _ in example.label_dict.values()]).any(axis=0)
        joint_usr_utt_label = np.array([x for _, x, _ in example.label_dict.values()]).any(axis=0)

        class_label_id_dict, start_pos_dict, end_pos_dict = {}, {}, {}
        for slot, (sys_utt_label, usr_utt_label, class_type) in example.label_dict.items():
            sys_tokens, sys_token_labels = \
                sub_tokenize(sys_utt_tokens, sys_utt_label, joint_sys_utt_label, tokenizer)
            usr_tokens, usr_token_labels = \
                sub_tokenize(usr_utt_tokens, usr_utt_label, joint_usr_utt_label, tokenizer)
            
            # TO DO : truncate
            token_labels = utils.pad(max_len, 0, [0] + sys_token_labels + [0] + usr_token_labels)
            class_label_id_dict[slot] = class_types.index(class_type)
            start_pos_dict[slot], end_pos_dict[slot] = \
                get_start_end_pos(class_type, token_labels, max_len)

        tokens = ['[CLS]'] + sys_tokens + ['[SEP]'] + usr_tokens + ['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0]*(2+len(sys_tokens)) + [1]*(1+len(usr_tokens))
        utils.pad(max_len, 0, input_ids, input_mask, segment_ids)

        return cls(example.guid, input_ids, input_mask, segment_ids, 
                    start_pos_dict, end_pos_dict, class_label_id_dict)


def get_start_end_pos(class_type, token_labels, max_len):
    if class_type == 'copy_value' and not any(token_labels):
        raise ValueError('copy_value but token_labels not detected.')
    if class_type != 'copy_value':
        return 0, 0 # start_pos, end_pos

    start_pos = token_labels.index(1)
    end_pos = max_len - 1 - token_labels[::-1].index(1)
    # tf.logging.info('token_label_ids: %s' % str(token_label_ids))
    # tf.logging.info('start_pos: %d' % start_pos)
    # tf.logging.info('end_pos: %d' % end_pos)
    assert all(token_labels[start_pos:end_pos+1])
    return start_pos, end_pos


def sub_tokenize(utt_tokens, utt_label, joint_utt_label, tokenizer, slot_value_dropout=0.3):
    tokens, token_labels = [], []
    for token, token_label, joint_label in zip(utt_tokens, utt_label, joint_utt_label):
        sub_tokens = tokenizer.tokenize(token)

        if slot_value_dropout == 0.0 or not joint_label:
            tokens.extend(sub_tokens)
        else:
            tokens.extend([sub_token if random.random() > slot_value_dropout else '[UNK]' \
                            for sub_token in sub_tokens])

        token_labels.extend([token_label for _ in sub_tokens])
    assert len(tokens) == len(token_labels)
    return tokens, token_labels



if __name__ == '__main__':
    examples = list(InputExample.read_woz('/ml/woz/woz_train_en.json', 'train'))
    
    for example in examples[0:2]:
        print(example)
        print()





