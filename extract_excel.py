"""
usage:
    python extract_excel.py [conf_path]
some problems:
    about extract excel:
        1. only for python3 environment
        2. if numbers of label column less than data column, keys of json is error
        3. if data is error originally, it can't correct the data.
        4. if more than one chart in one sheet, it will be meet some unknown error.
        5. some int type data would be extracted to float type
    about extract json:
        1. only for python3 environment
        2. it use word2vec, some the matched result is not correcting all the time.
            eg: if find word, '手机', there are two words in the label, '电话' and '邮箱',
                we hole to get '电话', but it return '邮箱'
        3. if one word includes another word, such as '推荐人电话' includes '电话', u must
        use '电话' as keyword not '电话号码' to avoid some error
        4. in this model, '身份证' is more similarity to '身份证地址' than '身份证号'
"""

import xlrd
import numpy as np
import pandas as pd
import json
import os
import sys
import gensim
import jieba
import configparser
import re
from tqdm import tqdm

FUZZY_MATCH = 1
ACCUATE_MATCH = 2


def extract_excel(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        for fn in os.listdir(output_dir):
            os.remove('%s/%s' % (output_dir, fn))

    for fn in tqdm(os.listdir(input_dir), file=sys.stdout):
        if os.path.splitext(fn)[-1] not in ['.xlsx', '.xls']:
            continue

        try:
            wb = xlrd.open_workbook(os.path.join(input_dir, fn))
        except Exception as e:
            sys.stderr.write(e.__repr__())
            continue

        for sn in wb.sheet_names():
            sheet = wb.sheet_by_name(sn)

            useless_col = []
            ncols = min(sheet.ncols, 100)
            for j in range(ncols):
                if all([a == '' for a in sheet.col_values(j)]):
                    useless_col.append(j)

            ras = []
            for i in range(sheet.nrows):
                num = 0
                for j in range(ncols):
                    if j in useless_col:
                        continue
                    if sheet.cell_value(i, j) != '':
                        num += 1
                ra = num / (sheet.ncols - len(useless_col) + 1)
                ras.append(ra)

            if len(ras) < 2:
                continue

            argmax = np.argmax(ras)
            if argmax > 10:
                sys.stderr.write('generate label %d error: %s_%s\n' % (argmax, fn, sn))
                continue
            label = sheet.row_values(argmax)

            with open(
                    os.path.join(output_dir, "%s_%s.json" % (os.path.splitext(fn)[0], re.sub(r'[\/:*?"<>|]', '', sn))),
                    'w', encoding='utf8') as f:
                for i in range(argmax + 1, sheet.nrows):
                    if ras[i] < 0.2:
                        continue
                    dic = {'file_name': fn, 'sheet_name': sn}
                    for j in range(sheet.ncols):
                        if j in useless_col:
                            continue
                        if label[j] in dic:
                            continue
                        cell = sheet.cell(i, j)
                        cell_value = cell.value
                        if cell.ctype in (2, 3) and int(cell_value) == cell_value:
                            cell_value = int(cell_value)
                        dic[label[j]] = str(cell_value)
                    try:
                        json.dump(dic, f, ensure_ascii=False, )
                        f.write('\n')
                    except UnicodeEncodeError:
                        sys.stderr.write('UnicodeEncodeError: %s_%s_%d\n' % (fn, sn, i))
                        continue


def extract_json(input_dir, output_path, primary_keys, second_keys, match_type=ACCUATE_MATCH):
    """
    :param primary_keys: 2-dim list, like [['keya1', 'keya2'], ['keyb1']], ues to sub repeat data
    :param second_keys: 2-dim list, like [['keya1', 'keya2'], ['keyb1']]
    :param input_dir:
    :param output_path:
    :return: return_dic, {index: bdic}
    """
    basic_dic = {k[0]: '' for k in primary_keys + second_keys}
    return_dic = {}
    pset = [set() for _ in primary_keys]  # list type, like [set(pkey1), set(pkey2)], ues to sub repeat data

    if match_type == 'fuzzy':
        model = gensim.models.Word2Vec.load('word2vec/sg.model')

    _type = os.path.splitext(output_path)[-1]
    if _type in ['.xls', '.xlsx']:
        output_type = 'excel'
    elif _type == '.json':
        output_type = 'json'
    else:
        output_type = 'csv'

    for fn in tqdm(os.listdir(input_dir), file=sys.stdout):
        with open(os.path.join(input_dir, fn), 'r', encoding='utf8') as f:
            match_dic = {}  # {the first string of keywords: matched string in file}, dict like: {'apple': 'apples'}
            for line in f:
                try:
                    js = json.loads(line, encoding='utf8')
                except json.decoder.JSONDecodeError:
                    sys.stderr.write('JSONDecodeError: %s\t%s\n' % (fn, line))
                    continue
                if len(match_dic) == 0:
                    for k in primary_keys + second_keys:
                        if match_type == 'fuzzy':
                            match_dic[k[0]] = fuzzy_match(model, k, js.keys())
                        elif match_type == 'accurate':
                            match_dic[k[0]] = accurate_match(k, js.keys())
                        else:
                            raise ValueError("mode not match!")

                if any([match_dic[k[0]] is not None
                        and match_dic[k[0]] in js
                        and js[match_dic[k[0]]] != ''
                        and js[match_dic[k[0]]] not in pset[i]
                        for i, k in enumerate(primary_keys)]):

                    [pset[i].add(js[match_dic[k[0]]]) for i, k in enumerate(primary_keys) if
                     match_dic[k[0]] is not None]
                    bdic = basic_dic.copy()
                    for k in primary_keys + second_keys:
                        if match_dic[k[0]] and match_dic[k[0]] in js:
                            bdic[k[0]] = js[match_dic[k[0]]]

                    return_dic[len(return_dic) + 1] = bdic

    if output_type == 'json':
        with open(output_path, 'w', encoding='utf8') as f:
            json.dump(return_dic, f, ensure_ascii=False, indent=4)
    else:
        data = []
        for v in return_dic.values():
            data.append([v[k[0]] for k in primary_keys + second_keys])
        c = [k[0] for k in primary_keys + second_keys]
        df = pd.DataFrame(data=data, columns=c)
        if output_type == 'excel':
            df.to_excel(output_path, index=None, encoding='utf8')
        elif output_type == 'csv':
            df.to_csv(output_path, sep='\t', index=None, encoding='utf8')

    return return_dic



# def fuzzy_match(model, keys, matched_keys, award_eps=1e-2, output_eps=.7):
#     matched_keys = list(matched_keys)
#     sims = []
#     args = []
#     for kk in keys:
#         sim = []
#         for k in matched_keys:
#             a = []
#             for w1 in jieba.cut(kk):
#                 if w1 not in model.wv:
#                     model.wv[w1] = np.random.random(128)
#                 for w in jieba.cut(k.replace(' ', '').replace('\u3000', '')):
#                     if w not in model.wv:
#                         model.wv[w] = np.random.random(128)
#                     a.append(model.similarity(w1, w))
#             if len(a) == 0:
#                 sim.append(0)
#                 continue
#             m = np.max(a)
#             mm = m
#             for aa in a:  # give it the awards and punishment
#                 if abs(aa - mm) < award_eps:
#                     m += 0.01
#                 else:
#                     m -= (1 - mm) * 0.001
#             sim.append(m)
#         args.append(np.argmax(sim))
#         sims.append(sim)
#
#     maxarg = np.argmax([sims[i][a] for i, a in enumerate(args)])
#     if sims[maxarg][args[maxarg]] > output_eps:
#         return matched_keys[args[maxarg]]



def sen2vec(model, W):
    v = np.zeros(128)
    cut = jieba.lcut(W)
    if len(cut) > 0:
        for w in cut:
            if w not in model.wv:
                model.wv[w] = np.random.random(128) / 100
            v += model.wv[w]
        v /= len(cut)
    return v


def fuzzy_match(model, keys, matched_keys, output_eps=2.1):
    matched_keys = list(matched_keys)
    sims = []
    args = []
    for kk in keys:
        sim = []
        v0 = sen2vec(model, kk)
        for k in matched_keys:
            v1 = sen2vec(model, k)
            sim.append(np.linalg.norm(v0 - v1))
        args.append(np.argmin(sim))
        sims.append(sim)
    argmin = np.argmin([sims[i][a] for i, a in enumerate(args)])
    if sims[argmin][args[argmin]] < output_eps:
        return matched_keys[args[argmin]]


def accurate_match(keys, matched_keys):
    for k in keys:
        if k in matched_keys:
            return k


if __name__ == '__main__':
    if len(sys.argv) == 1:
        conf_path = 'extract_excel.conf'
    else:
        conf_path = sys.argv[1]

    conf = configparser.ConfigParser()
    conf.read(conf_path, encoding='utf8')

    # sys.stderr = open(conf.get('DEFAULT', 'log_file'), 'w', encoding='utf8')

    if conf.get('DEFAULT', 'extract_json') == 'true':
        extract_excel(conf.get('DEFAULT', 'input_dir'),
                      conf.get('DEFAULT', 'output_json_dir'))

    if conf.get('DEFAULT', 'extract_excel') == 'true':
        extract_json(conf.get('DEFAULT', 'output_json_dir'),
                     conf.get('DEFAULT', 'output_excel_file'),
                     [k.split(':') for k in conf.get('DEFAULT', 'primary_keywords').split(' ')],
                     [k.split(':') for k in conf.get('DEFAULT', 'keywords').split(' ')],
                     conf.get('DEFAULT', 'match_mode'))