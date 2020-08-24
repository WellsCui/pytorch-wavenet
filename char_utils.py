
from typing import Callable, Any, Iterable, List
import numpy as np


class CharacterDic(object):
    def __init__(self):
        self.chars = ' abcdefghijklmnopqrstuvwxyz0123456789,.?\''
        self.char_to_index_map = {}
        for i, c in enumerate(self.chars):
            self.char_to_index_map[c] = i
    
    def str2Idx(self, sents: List[str]) -> (List[List[int]], List[int]):
        rs = []
        lengths = [len(sent) for sent in sents]
        max_len = np.amax(lengths)
        for i, sent in enumerate(sents):
            idxs = [self.char_to_index_map.get(c, 41) for c in sent.lower()]+[0]*(max_len-lengths[i])
            rs.append(idxs)
        return np.array(rs), lengths
