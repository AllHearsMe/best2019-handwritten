import numpy as np
from itertools import groupby

componentList = ['consonant', 'vowel', 'tone']
componentIdx = {s: i for i, s in enumerate(componentList)}
componentCount = len(componentList)

blankChar = '-'

charList = [None]*componentCount
charList[componentIdx['consonant']
         ] = '0123456789กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮฯๆะาำเแโใไฤฦๅ'
charList[componentIdx['vowel']] = 'ัิีึืุูํ็'
charList[componentIdx['tone']] = '่้๊๋์'
charList = [' ' + l + blankChar for l in charList]
charListAll = ''.join(charList)

charIdx = [{ch: i for i, ch in enumerate(
    charList[k])} for k in range(componentCount)]
charSet = [set(charList[k]) for k in range(componentCount)]
charCount = [len(charList[k]) for k in range(componentCount)]
toOnehot = [np.eye(charCount[k]) for k in range(componentCount)]
toOnehotAll = np.eye(sum(charCount))

charType = {ch: k for k in range(componentCount) for ch in charList[k]}
charType[blankChar] = -1
charType[' '] = 0

onehotOffsetEnd = list(np.cumsum(charCount))
onehotOffset = [0]+onehotOffsetEnd[:-1]
onehotLen = onehotOffsetEnd[-1]
onehotSlices = [slice(s, e) for (s, e) in zip(onehotOffset, onehotOffsetEnd)]
onehotIdx = {ch: (charIdx[t][ch] + onehotOffset[t])
             for t in range(componentCount) for ch in charList[t]}
onehotIdx[' '] = 0
onehotIdx[blankChar] = onehotLen - 1


def str2idx(s, length=0, split=True):
    if split:
        idx = []
        for ch in s:
            cmp = charType[ch]
            if cmp < 0:
                continue
            elif not idx or cmp == 0 or idx[-1][cmp] >= 0:
                idx.append([-1]*3)
            idx[-1][cmp] = charIdx[cmp][ch]
        idx = np.maximum(idx, 0)
        return np.pad(idx, ((0, max(0, length - len(idx))), (0, 0)), 'constant')
    else:
        idx = [onehotIdx[ch] for ch in s]
        return np.pad(idx, ((0, max(0, length - len(idx)))), 'constant')
    

def idx2str(idx, split=True):
    if split:
        n = idx.shape[0]
        s = [charList[j][idx[i][j]]
             for i in range(n) for j in range(componentCount)
             if (charList[j][idx[i][j]] != blankChar)
             and ((j == 0) or (charList[j][idx[i][j]] != ' '))]
    else:
        s = [charListAll[x] for x in idx if charListAll[x] != blankChar]
    return ''.join(s)


def idx2onehot(idx, split=True):
    if split:
        l = [toOnehot[i][idx[:, i]] for i in range(componentCount)]
        onehot = np.concatenate(l, axis=-1)
    else:
        onehot = toOnehotAll[idx]
    return onehot


def onehot2idx(onehot, split=True):
    if split:
        l = [np.argmax(onehot[..., sl], axis=-1)[..., None]
             for sl in onehotSlices]
        idx = np.concatenate(l, axis=-1)
    else:
        idx = np.argmax(onehot, axis=-1)
    return idx


def str2onehot(s, length=0, split=True):
    onehot = idx2onehot(str2idx(s, split=split), split=split)
    return np.pad(onehot, ((0, max(0, length - onehot.shape[0])), (0, 0)), 'constant')


def onehot2str(onehot, split=True):
    return idx2str(onehot2idx(onehot, split=split), split=split)

def collapse(idx, split=True):
    if split:
        idxs = [[k for k, _ in groupby(i) if k < charCount[tp]-1] for tp, i in enumerate(zip(*idx))]
        maxlen = max([len(i) for i in idxs])
        idxs = [np.pad(i, ((0, max(0, maxlen - len(i)))), 'constant').astype(int) for i in idxs]
        return np.stack(idxs, axis=-1)
    else:
        return [k for k, _ in groupby(idx) if k < onehotLen-1]

def decode(idx, split=True):
    return idx2str(collapse(idx, split), split)


if __name__ == '__main__':
    onehot = str2onehot('เป็นมนุษย์สุดประเสริฐเลิศคุณค่า', split=True)
    print(onehot.shape)
    s = onehot2str(onehot, split=True)
    print(s)
    onehot = str2onehot('เป็นมนุษย์สุดประเสริฐเลิศคุณค่า', split=False)
    print(onehot.shape)
    s = onehot2str(onehot, split=False)
    print(s)
