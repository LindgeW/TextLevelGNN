from collections import Counter, OrderedDict
from functools import wraps
import numpy as np
from transformers import BertTokenizer


def _check_build_bert_vocab(func):
    @wraps(func)
    def _wrapper(self, *args, **kwargs):
        if self.bert_tokenizer is None:
            self.build_vocab()
        return func(self, *args, **kwargs)

    return _wrapper


class BERTVocab(object):
    def __init__(self, bert_vocab_path=None):
        self.bert_tokenizer = None
        self.bert_vocab_path = bert_vocab_path

    def build_vocab(self):
        if self.bert_tokenizer is None and self.bert_vocab_path is not None:
            self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_path)
            print("Load bert vocabulary finished !!!")
        return self

    @_check_build_bert_vocab
    def state_dict(self):
        state = OrderedDict()
        state_attrs = ['bert_tokenizer', 'bert_vocab_path']
        for attr in state_attrs:
            if hasattr(self, attr):
                state[attr] = getattr(self, attr)
        return state

    @classmethod
    def load_state_dict(cls, state_dict):
        new_ = cls()
        for attr, val in state_dict.items():
            setattr(new_, attr, val)
        return new_

    @_check_build_bert_vocab
    def bert2id(self, tokens: list):
        '''将原始token序列转换成bert bep ids'''
        def transform(token):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))

        tokenizer = self.bert_tokenizer
        cls_tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        bert_ids = transform(' '.join(cls_tokens))
        # bert_ids = tokenizer.encode(' '.join(tokens), add_special_tokens=True)  # 传入sentence
        segment = [0] * len(bert_ids)
        bert_mask = [1] * len(bert_ids)
        return bert_ids, segment, bert_mask

    @_check_build_bert_vocab
    def batch_bert2id(self, tokens_lst: list):
        # '''将原始token序列转换成bert bep ids'''
        # def transform(token):
        #     return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))

        bert_ids, segments, bert_masks = [], [], []
        tokenizer = self.bert_tokenizer
        max_len = 0
        for tokens in tokens_lst:
            # cls_tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
            # token_ids = transform(' '.join(cls_tokens))
            token_ids = tokenizer.encode(' '.join(tokens), add_special_tokens=True)
            bert_ids.append(token_ids)
            segments.append([0] * len(token_ids))
            bert_masks.append([1] * len(token_ids))
            if len(token_ids) > max_len:
                max_len = len(token_ids)

        for i in range(len(bert_ids)):
            padding = [0] * (max_len - len(bert_ids[i]))
            bert_ids[i] += padding
            segments[i] += padding
            bert_masks[i] += padding

        return bert_ids, segments, bert_masks

    @_check_build_bert_vocab
    def bertwd2id(self, tokens: list):
        '''将原始token序列转换成bert bep ids'''
        def transform(token):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))

        tokenizer = self.bert_tokenizer
        cls_tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        bert_ids, bert_len = [], []
        bert_piece_ids = map(transform, cls_tokens)
        for piece in bert_piece_ids:
            if not piece:
                piece = transform(tokenizer.unk_token)
            bert_ids.extend(piece)
            bert_len.append(len(piece))

        segment = [0] * len(bert_ids)
        bert_mask = [1] * len(bert_ids)
        return bert_ids, segment, bert_mask, bert_len

    @_check_build_bert_vocab
    def batch_bertwd2id(self, tokens_lst: list):
        '''将原始token序列转换成bert bep ids'''
        def transform(token):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))

        bert_ids, segments, bert_masks, bert_lens = [], [], [], []
        tokenizer = self.bert_tokenizer
        for tokens in tokens_lst:
            cls_tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
            bert_piece_ids = map(transform, cls_tokens)
            bert_id, bert_len = [], []
            for piece in bert_piece_ids:
                if not piece:
                    piece = transform(tokenizer.unk_token)
                bert_id.extend(piece)
                bert_len.append(len(piece))

            bert_ids.append(bert_id)
            segments.append([0] * len(bert_id))
            bert_masks.append([1] * len(bert_id))
            bert_lens.append(bert_len)

        max_bpe_len = max(len(x) for x in bert_ids)
        max_seq_len = max(len(y) for y in bert_lens)

        for i in range(len(bert_ids)):
            padding = [0] * (max_bpe_len - len(bert_ids[i]))
            bert_ids[i] += padding
            segments[i] += padding
            bert_masks[i] += padding

        for j in range(len(bert_lens)):
            padding = [0] * (max_seq_len - len(bert_lens[j]))
            bert_lens[j] += padding

        return bert_ids, segments, bert_masks, bert_lens


# 检查词表是否被创建
def _check_build_vocab(func):
    @wraps(func)
    def _wrapper(self, *args, **kwargs):
        if self._inst2idx is None:
            self.build_vocab()
        return func(self, *args, **kwargs)
    return _wrapper


class Vocab(object):
    def __init__(self, min_count=None, lower=False,
                 pad='<pad>', unk='<unk>',
                 bos='<bos>', eos='<eos>'):
        self.min_count = min_count
        self._idx2inst = None
        self._inst2idx = None
        self._inst_count = Counter()
        self.embeddings = None

        self.lower = lower
        self.PAD = pad
        self.UNK = unk
        self.BOS = bos
        self.EOS = eos

    def add(self, inst):
        if isinstance(inst, list):
            if self.lower:
                inst = [i.lower() for i in inst if isinstance(i, str)]
            self._inst_count.update(inst)
        else:
            if self.lower and isinstance(inst, str):
                inst = inst.lower()
            self._inst_count[inst] += 1

    def build_vocab(self):
        if self._inst2idx is None:
            self._inst2idx = dict()
            if self.PAD is not None:
                self._inst2idx[self.PAD] = len(self._inst2idx)
            if self.UNK is not None:
                self._inst2idx[self.UNK] = len(self._inst2idx)
            if self.BOS is not None:
                self._inst2idx[self.BOS] = len(self._inst2idx)
            if self.EOS is not None:
                self._inst2idx[self.EOS] = len(self._inst2idx)

        min_count = 1 if self.min_count is None else self.min_count
        for inst, count in self._inst_count.items():
            if count >= min_count and inst not in self._inst2idx:
                self._inst2idx[inst] = len(self._inst2idx)

        self._idx2inst = dict((idx, inst) for inst, idx in self._inst2idx.items())
        # del self._inst_count
        return self

    # def load_embeddings(self, embed_path):
    #     vocab_size = len(self)
    #     nb_embeddings = 0
    #     with open(embed_path, 'r', encoding='utf-8') as fin:
    #         for line in fin:
    #             tokens = line.strip().split(' ')
    #             if len(tokens) < 10:
    #                 continue
    #
    #             idx = self._inst2idx.get(tokens[0])
    #             if idx is not None:
    #                 vec = np.asarray(tokens[1:], dtype=np.float32)
    #                 if self.embeddings is None:
    #                     vec_size = len(vec)
    #                     # self.embeddings = np.random.randn(vocab_size, vec_size).astype(np.float32)
    #                     self.embeddings = np.zeros((vocab_size, vec_size), np.float32)
    #                     self.embeddings[self.pad_idx] = 0
    #                 self.embeddings[idx] = vec
    #                 self.embeddings[self.unk_idx] += vec
    #                 nb_embeddings += 1
    #
    #     self.embeddings[self.unk_idx] /= nb_embeddings
    #     self.embeddings /= np.std(self.embeddings)
    #     return nb_embeddings

    def load_embeddings(self, embed_path):
        vocab_size = len(self)
        nb_embeddings = 0
        with open(embed_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                tokens = line.strip().split(' ')
                if len(tokens) < 10:
                    continue

                if tokens[0] in self._inst2idx:
                    if self.embeddings is None:
                        vec_size = len(tokens[1:])
                        self.embeddings = np.random.uniform(-0.5/vec_size, 0.5/vec_size, (vocab_size, vec_size)).astype(np.float32)
                    vec = np.asarray(tokens[1:], dtype=np.float32)
                    self.embeddings[self._inst2idx[tokens[0]]] = vec
                    # self.embeddings[self.unk_idx] += vec
                    nb_embeddings += 1
        self.embeddings[self.pad_idx] = 0.
        # self.embeddings[self.unk_idx] /= nb_embeddings     # nb_embeddings较大时有损性能
        self.embeddings /= np.std(self.embeddings)
        return nb_embeddings

    @_check_build_vocab
    def inst2idx(self, inst):
        if isinstance(inst, list):
            return [self._inst2idx.get(i, self.unk_idx) for i in inst]
        else:
            return self._inst2idx.get(inst, self.unk_idx)

    @_check_build_vocab
    def idx2inst(self, idx):
        if isinstance(idx, list):
            res = []
            for i in idx:
                tag = self._idx2inst.get(i, self.UNK)
                if tag != self.EOS:
                    res.append(tag)
                else:
                    break
            return res
        else:
            return self._idx2inst.get(idx, self.UNK)

    @_check_build_vocab
    def state_dict(self):
        state = OrderedDict()
        state_attrs = ['_inst2idx', '_idx2inst', 'PAD', 'UNK', 'BOS', 'EOS']
        for attr in state_attrs:
            if hasattr(self, attr):
                state[attr] = getattr(self, attr)
        return state

    @classmethod
    def load_state_dict(cls, state_dict):
        new_ = cls()
        for attr, val in state_dict.items():
            setattr(new_, attr, val)
        return new_

    @property
    @_check_build_vocab
    def pad_idx(self):
        if self.PAD is None:
            return None
        return self._inst2idx[self.PAD]

    @property
    @_check_build_vocab
    def unk_idx(self):
        if self.UNK is None:
            return None
        return self._inst2idx[self.UNK]

    @property
    @_check_build_vocab
    def bos_idx(self):
        if self.BOS is None:
            return None
        return self._inst2idx[self.BOS]

    @property
    @_check_build_vocab
    def eos_idx(self):
        if self.EOS is None:
            return None
        return self._inst2idx[self.EOS]

    @_check_build_vocab
    def __len__(self):
        return len(self._inst2idx)

    @_check_build_vocab
    def __iter__(self):
        for inst, idx in self._inst2idx.items():
            yield inst, idx

    @_check_build_vocab
    def __contains__(self, item):
        return item in self._inst2idx


class MultiVocab(object):
    '''
        Vocab container
    '''
    def __init__(self, vocab_dict=None):
        self._vocabs = OrderedDict()  # 有序的dict结构

        if vocab_dict is None:
            return

        for k, v in vocab_dict.items():
            self._vocabs[k] = v

    def __setitem__(self, key, item):
        self._vocabs[key] = item

    def __getitem__(self, key):
        return self._vocabs[key]

    def state_dict(self):
        state = OrderedDict()
        key2class = OrderedDict()
        for k, v in self._vocabs.items():
            state[k] = v.state_dict()
            # vocab name to class name mapping
            key2class[k] = type(v).__name__
        state['_key2class_'] = key2class
        return state

    @classmethod
    def load_state_dict(cls, state_dict):
        assert '_key2class_' in state_dict
        cls_dict = {'BERTVocab': BERTVocab, 'Vocab': Vocab}
        new_ = cls()
        key2class = state_dict.pop('_key2class_')
        for k, v in state_dict.items():
            cls_name = key2class[k]
            new_[k] = cls_dict[cls_name].load_state_dict(v)
        return new_


if __name__ == '__main__':
    bert_vocab = BERTVocab('../bert/en_bert/vocab.txt')
    bert_ids1 = bert_vocab.bert2id('I love you'.split(' '))[0]
    print(bert_ids1)
    bert_ids2 = bert_vocab.bert2id('i Love you'.split(' '))[0]
    print(bert_ids2)