


import json

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

class InputExample(object):
    def __init__(self, guid, text_a, text_b, 
                 text_a_label=None, text_b_label=None, class_label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_a_label = text_a_label
        self.text_b_label = text_b_label
        self.class_label = class_label
    
    def __str__(self):
        return "\ntext_a :" + str(self.text_a) \
            + "\ntext_b :" + str(self.text_b) \
            + "\ntext_a_label :" + str(self.text_a_label) \
            + "\ntext_b_label :" + str(self.text_b_label) \
            + "\nclass_label :" + str(self.class_label)

    @staticmethod
    def fix(label):
        fix_table = {'centre': 'center', 'areas': 'area', 'phone number': 'number'}
        return fix_table.get(label.strip(), label.strip())

    @classmethod
    def read(cls, filepath, slot_list, set_type='train'):
        for dialogue in json.load(open(filepath)):
            for turn in dialogue['dialogue']:
                guid = '%s-%s-%s' % (set_type,
                                    str(dialogue['dialogue_idx']),
                                    str(turn['turn_idx']))
                print(guid)

                sys_utt_tokens = utils.tokenize(turn['system_transcript'])
                usr_utt_tokens = utils.tokenize(turn['transcript'])

                #print(sys_utt_words)
                #print(usr_utt_words)

                # {slot: value}
                turn_label = {cls.fix(s): cls.fix(v) for s, v in turn['turn_label']}
                print(turn_label)

                sys_utt_tok_label_dict, usr_utt_tok_label_dict = {}, {}
                for slot in slot_list:
                    label = turn_label.get(slot, 'none')

                    


                print()
                
            #break


def find_label_pos(tokens, label):
    positions = []
    label_tokens = utils.tokenize(label)
    for i in (i for i, e in enumerate(tokens) if e == label_tokens[0]):
        if tokens[i:i+len(label_tokens)] == label_tokens:
            positions.append((i, i+len(label_tokens))) # start, exclusive_end
    return positions


def check_label_existence(label, usr_utt_tokens, sys_utt_tokens):
    usr_pos = find_label_pos(usr_utt_tokens, label)
    sys_pos = find_label_pos(sys_utt_tokens, label)

    if len(usr_pos) == 0 and len(sys_pos) == 0 and label in SEMANTIC_DICT:
        for label_substitute in SEMANTIC_DICT[label]:
            usr_pos = find_label_pos(usr_utt_tokens, label_substitute)
            sys_pos = find_label_pos(sys_utt_tokens, label_substitute)
            if len(usr_pos) > 0 or len(sys_pos) > 0:
                label = label_substitute
                break
    return label, usr_pos, sys_pos


def get_turn_label(label, sys_utt_tok, usr_utt_tok, slot_last_occurrence):
  sys_utt_tok_label = [0 for _ in sys_utt_tok]
  usr_utt_tok_label = [0 for _ in usr_utt_tok]
  if label == 'none' or label == 'dontcare':
      class_type = label
  else:
    label, in_usr, usr_pos, in_sys, sys_pos = check_label_existence(label, usr_utt_tok, sys_utt_tok)
    if in_usr or in_sys:
      class_type = 'copy_value'
      if slot_last_occurrence:
        if in_usr:
          (s, e) = usr_pos[-1]
          for i in range(s, e):
            usr_utt_tok_label[i] = 1
        else:
          (s, e) = sys_pos[-1]
          for i in range(s, e):
            sys_utt_tok_label[i] = 1
      else:
        for (s, e) in usr_pos:
          for i in range(s, e):
            usr_utt_tok_label[i] = 1
        for (s, e) in sys_pos:
          for i in range(s, e):
            sys_utt_tok_label[i] = 1
    else:
      class_type = 'unpointable'
  return sys_utt_tok_label, usr_utt_tok_label, class_type




if __name__ == '__main__':
    class_types = ['none', 'dontcare', 'copy_value', 'unpointable']
    slot_list = ['area', 'food', 'price range']

    examples = InputExample.read('/ml/woz/woz_train_en.json', slot_list, 'train')





