import codecs
import pandas as pd
import os
from tqdm import tqdm
import zipfile
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[ %(asctime)s ] %(message)s'
)


def ignore_errors(error=Exception):
    def wrap2(func):
        def wrap(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error as e:
                logging.error(e)

        return wrap

    return wrap2


def get_df(obj, **kwargs):
    """返回一个读取df的迭代器

    Parameters
    ----------
    obj:
        str: 文件名或者文件路径
        ZipFile: 压缩文件路径
        list: obj列表
        df: pd.DataFrame
    kwargs:
        读文件时可选的传入参数

    Returns
    -------
    生成器:
        (df, 文件名/第n个obj)

    Examples
    --------
    >>> for df, fn in get_df(['1.xlsx', 'data']):
    ...     print(fn)
    1.xlsx
    data\\1.xlsx
    data\\2.txt
    data\\3.zip\\1.xlsx
    data\\3.zip\\新建/1.xlsx
    data\\3.zip\\4.zip\\1.xlsx
    """
    # 文件名或者文件路径
    if isinstance(obj, str):
        # 如果是文件名，则判断其文件是否为Excel文件后缀或者是zip文件后缀，否则都当做是csv文件读取，如果读取失败则跳过
        if os.path.isfile(obj):
            try:
                if obj.endswith('.xlsx') or obj.endswith('.xls'):
                    yield pd.read_excel(obj, **kwargs), obj
                elif obj.endswith('.zip'):
                    for _ in get_df(zipfile.ZipFile(obj), **kwargs):
                        yield _
                else:
                    kwargs.pop('n_items')
                    yield pd.read_csv(obj, encoding='utf8', **kwargs), obj
            except Exception as e:
                logging.error('%s: %s' % (obj, e))

        # 如果是文件夹路径，则遍历文件夹下文件，递归调取该函数
        elif os.path.isdir(obj):
            for file in os.listdir(obj):
                for _ in get_df(os.path.join(obj, file), **kwargs):
                    yield _

        else:
            logging.error('%s: %s' % (obj, 'No such file or director!'))

    # 压缩文件格式
    elif isinstance(obj, zipfile.ZipFile):
        for fn in obj.namelist():
            try:
                if fn.endswith('.xlsx') or fn.endswith('.xls'):
                    yield pd.read_excel(obj.open(fn), **kwargs), \
                          os.path.join(kwargs.get('top_fn', ''), obj.filename,
                                       fn.encode('cp437').decode('gbk'))  # 中文文件正常显示
                elif fn.endswith('.zip'):
                    for _ in get_df(zipfile.ZipFile(obj.open(fn)), top_fn=obj.filename, **kwargs):
                        yield _
                elif not fn.endswith('/'):
                    yield pd.read_csv(obj.open(fn), encoding='utf8', **kwargs), \
                          os.path.join(kwargs.get('top_fn', ''), obj.filename, fn.encode('cp437').decode('gbk'))
            except Exception as e:
                logging.error('%s: %s' % (fn, e))

        obj.close()

    # df列表，则递归调用该函数
    elif isinstance(obj, list):
        for i, obj_ in enumerate(obj):
            for _ in get_df(obj_, n_items=i, **kwargs):
                yield _

    # 如果是df，则直接返回即可
    elif isinstance(obj, pd.DataFrame):
        yield obj, kwargs.get('n_items', -1)

    else:
        logging.error('%s: %s' % (obj, 'Input Type is not supported!'))


def save_dfs(dfs, save_type='one-excel', tag_columns=None, save_tags=None, save_path=''):
    """将多个df按照每个df的标签，保存成一个或多个excel文件

    Parameters
    ----------
    dfs:
        一维df列表，[df1, df2, ...]
    save_type:
        one-excel：
            存储在同一个excel中，每一个tag保存为一张sheet表，每个sheet表的名字为 `save_tags`，
            如果 `save_tags` 没有设置，则采用默认值 sheet1, sheet2, ..., sheetn
        one-sheet：
            存储在同一个excel的同一张sheet表中，会增加一个新的列 `tag_columns`，来区别不同的tag
            如果 `save_tags` 没有设置，则采用默认值 tag1, tag2, ..., tagn
        multi-excel：
            每一个 tag 存储为一个excel表，文件名为 `save_tags`
            如果 `save_tags` 没有设置，则采用默认值 file1, file2, ..., filen
        其他情况：
            不做任何处理
    tag_columns:
        用于 save_type='one-sheet' 的时候，tag保存时的索引，默认是 `tag_name`
    save_tags:
        每个标签保存的名字
    save_path:
        保存路径，如果 save_type='one-excel'/'one-sheet' 则为文件路径，如果 save_type='multi-excel' 则为文件夹路径
    """
    # 每一个tag保存为一张sheet表
    if save_type == 'one-excel':
        save_tags = save_tags or ['sheet%d' % i for i in range(len(dfs))]
        # 将url存储为字符串格式，防止url达到65535条上限报错
        with pd.ExcelWriter(save_path, engine='xlsxwriter', options={'strings_to_urls': False}) as fw:
            for i, df in enumerate(dfs):
                df.to_excel(fw, index=False, sheet_name=save_tags[i])

    # 所有数据存储在一张sheet表中
    elif save_type == 'one-sheet':
        save_tags = save_tags or ['tag%d' % i for i in range(len(dfs))]
        for i, df in enumerate(dfs):
            df[tag_columns or 'tag_name'] = [save_tags[i]] * len(df)

        df = pd.concat(dfs, ignore_index=True)
        df.to_excel(save_path, index=False)

    # 每一个tag存储为一个excel表
    elif save_type == 'multi-excel':
        save_tags = save_tags or ['file%d.xlsx' % i for i in range(len(dfs))]
        save_paths = [os.path.join(save_path, i) for i in save_tags]

        for i, df in enumerate(dfs):
            df.to_excel(save_paths[i], index=False)


def concat_dfs(dfs, drop_duplicates_columns=None, attach_index=None):
    """将多个df合并成一个df

    Parameters
    ----------
    dfs:
        一维df列表，[df1, df2, ..., dfn]
    drop_duplicates_columns:
        需要去重的标签
    attach_index:
        dict type，额外添加多个列，{col1: value1, col2: value2, ..., coln: valuen}

    Returns
    -------
    DataFrame:
        合并后的df
    """
    dfs_concat = []
    for i, _ in enumerate(dfs):
        if not _:  # 空集
            dfs_concat.append(pd.DataFrame())
            continue

        df = pd.concat(_, ignore_index=True)

        if drop_duplicates_columns:
            df = df.drop_duplicates(subset=drop_duplicates_columns)

        if attach_index:
            for k, v in attach_index.items():
                df[k] = [v[i]] * len(df)

        dfs_concat.append(df)

    return dfs_concat


def union_file(obj, rename_columns=None,
               tags=None, drop_duplicates_columns=None, attach_index=None,
               save_type='one-excel', tag_columns=None, save_tags=None, save_path='', **kwargs):
    """将多个csv/excel文件按照文件名中带有不同的tag，相同的tag合并，然后保存为excel格式
    如果传入的是excel，只会读取第一个sheet表，如果想要读取所有的sheet表，调用union_excel()方法

    Parameters
    ----------
    obj:
        str: 文件名或者文件路径
        ZipFile: 压缩文件路径
        list: obj列表
        df: pd.DataFrame
    rename_columns:
        字段重命名，如果不设置，则才有原有的字段
    tags:
        文件分类标签，把文件名带有同一个tag的归为一类，如果不指定，则将所有df归为一类
    drop_duplicates_columns:
        需要去重的标签
    attach_index:
        dict type，额外添加多个列，{col1: value1, col2: value2, ..., coln: valuen}
    save_type:
        one-excel：
            存储在同一个excel中，每一个tag保存为一张sheet表，每个sheet表的名字为 `save_tags`，
            如果 `save_tags` 没有设置，则采用默认值 sheet1, sheet2, ..., sheetn
        one-sheet：
            存储在同一个excel的同一张sheet表中，会增加一个新的列 `tag_columns`，来区别不同的tag
            如果 `save_tags` 没有设置，则采用默认值 tag1, tag2, ..., tagn
        multi-excel：
            每一个 tag 存储为一个excel表，文件名为 `save_tags`
            如果 `save_tags` 没有设置，则采用默认值 file1, file2, ..., filen
        其他情况：
            返回一个合并后的df
    tag_columns:
        用于 save_type='one-sheet' 的时候，tag保存时的索引，默认是 `tag_name`
    save_tags:
        每个标签保存的名字
    save_path:
        保存路径，如果 save_type='one-excel'/'one-sheet' 则为文件路径，如果 save_type='multi-excel' 则为文件夹路径
    kwargs:
        读文件时可选的传入参数

    Returns
    -------
    list:
        一维df列表

    Examples
    ---------
    >>> os.listdir('data')
    ['test1.xlsx', 'test2.xlsx', 'train1.xlsx', 'train2.xlsx']
    >>> dfs = union_file('data', tags=['test', 'train'], save_path='union_file.xlsx')
    >>> for df in dfs:
    ...     print(df)
         col
    0  test1
    1  test2
          col
    0  train1
    1  train2
    >>> df_dict = pd.read_excel('union_file.xlsx', sheet_name=None)
    >>> for k, v in df_dict.items():
    ...     print(k)
    ...     print(v)
    test
         col
    0  test1
    1  test2
    train
          col
    0  train1
    1  train2
    """
    if tags:
        dfs = [[] for _ in range(len(tags))]
    else:
        dfs = [[]]

    for _ in tqdm(get_df(obj, **kwargs)):
        if not _:
            continue

        df, fn = _

        if rename_columns:
            df.columns = rename_columns

        if tags:
            for i, tag in enumerate(tags):
                if tag in fn:
                    dfs[i].append(df)
        else:
            dfs[0].append(df)

    dfs_concat = concat_dfs(dfs, drop_duplicates_columns, attach_index)

    # 如果没有定义保存的tags，就用分类的tags
    if not save_tags:
        save_tags = tags

    save_dfs(dfs_concat, save_type, tag_columns, save_tags, save_path)

    return dfs_concat


def union_excel(obj, read_all_sheet=True, rename_columns=None,
                tags=None, drop_duplicates_columns=None, attach_index=None,
                save_type='one-excel', tag_columns=None, save_tags=None, save_path='', **kwargs):
    """如果 read_all_sheet=False，则直接调用 union_file 函数，
    如果 read_all_sheet=True，则将多个excel文件按照相同的列名合并成一个excel，只能传入excel文件名
    该函数的参数与 union_file 函数类似，其他参数使用方法请参考 union_file 函数"""

    if not read_all_sheet:
        return union_file(obj, rename_columns,
                          tags, drop_duplicates_columns, attach_index,
                          save_type, tag_columns, save_tags, save_path, **kwargs)

    dfs = []
    dfs_dic = {}

    for _ in tqdm(get_df(obj, sheet_name=None, **kwargs)):
        if _ is None:
            continue

        df_dict, fn = _

        for key, df in df_dict.items():
            if rename_columns:
                df.columns = rename_columns

            if tags:
                for t in tags:
                    if t in key:
                        df_list = dfs_dic.get(t, [])
                        df_list.append(df)
                        dfs_dic[t] = df_list
            else:
                df_list = dfs_dic.get(key, [])
                df_list.append(df)
                dfs_dic[key] = df_list

    if not tags:
        tags = list(dfs_dic.keys())

    for t in tags:
        dfs.append(dfs_dic[t])

    if not save_tags:
        save_tags = list(dfs_dic.keys())

    dfs_concat = concat_dfs(dfs, drop_duplicates_columns, attach_index)

    if not save_tags:
        save_tags = tags

    save_dfs(dfs_concat, save_type, tag_columns, save_tags, save_path)

    return dfs_concat


def split_file(obj, split_columns, rename_columns=None,
               tags=None, drop_duplicates_columns=None, attach_index=None,
               save_type='one-excel', tag_columns=None, save_path='', **kwargs):
    """将多个csv/excel文件某一列带有不同的tag，拆分成多个df，带有相同tag的df合并，然后分别保存为excel格式
    如果传入的是excel，只会读取第一个sheet表，如果想要读取所有的sheet表，调用split_excel()方法

    Parameters
    ----------
    obj:
        str: 文件名或者文件路径
        ZipFile: 压缩文件路径
        list: obj列表
        df: pd.DataFrame
    split_columns:
        拆分依据列
    rename_columns:
        字段重命名，如果不设置，则才有原有的字段
    tags:
        文件分类标签，把文件名带有同一个tag的归为一类，如果不指定，则按分组划分
    drop_duplicates_columns:
        需要去重的标签
    attach_index:
        dict type，额外添加多个列，{col1: value1, col2: value2, ..., coln: valuen}
    save_type:
        one-excel：
            存储在同一个excel中，每一个tag保存为一张sheet表，每个sheet表的名字为 `save_tags`，
            如果 `save_tags` 没有设置，则采用默认值 sheet1, sheet2, ..., sheetn
        one-sheet：
            存储在同一个excel的同一张sheet表中，会增加一个新的列 `tag_columns`，来区别不同的tag
            如果 `save_tags` 没有设置，则采用默认值 tag1, tag2, ..., tagn
        multi-excel：
            每一个 tag 存储为一个excel表，文件名为 `save_tags`
            如果 `save_tags` 没有设置，则采用默认值 file1, file2, ..., filen
        其他情况：
            返回一个合并后的df
    tag_columns:
        用于 save_type='one-sheet' 的时候，tag保存时的索引，默认是 `tag_name`
    save_path:
        保存路径，如果 save_type='one-excel'/'one-sheet' 则为文件路径，如果 save_type='multi-excel' 则为文件夹路径
    kwargs:
        读文件时可选的传入参数

    Returns
    -------
    list, list:
        一维df列表，一维 save_tags 列表

    Examples
    ---------
    >>> pd.read_excel('test.xlsx')
       values tags
    0       1   a1
    1       2   a2
    2       3   b1
    3       4   b2
    >>> dfs_concat, save_tags = split_file('test.xlsx', split_columns='tags', tags=['a', 'b'], save_path='split_file.xlsx')
    >>> save_tags
    ['a', 'b']
    >>> for df in dfs_concat:
    ...     print(df)
       values tags
    0       1   a1
    1       2   a2
       values tags
    0       3   b1
    1       4   b2
    >>> df_dict = pd.read_excel('split_file.xlsx', sheet_name=None)
    >>> for k, v in df_dict.items():
    ...     print(k)
    ...     print(v)
    a
       values tags
    0       1   a1
    1       2   a2
    b
       values tags
    0       3   b1
    1       4   b2
    """

    tag_df = {}
    for _ in tqdm(get_df(obj, **kwargs)):
        if _ is None:
            continue

        df, fn = _

        if rename_columns:
            df.columns = rename_columns

        if tags:
            tag_index = {tag: [] for tag in tags}
            for index, row in df.iterrows():
                for tag in tags:
                    if tag in row[split_columns]:
                        tag_index[tag].append(index)

            for tag, indexes in tag_index.items():
                dfs = tag_df.get(tag, [])
                dfs.append(df.loc[indexes])
                tag_df[tag] = dfs

        # 如果没指定tag，就按分组划分
        else:
            for tag, df_ in df.groupby(split_columns):
                dfs = tag_df.get(tag, [])
                dfs.append(df_)
                tag_df[tag] = dfs

    save_tags, dfs = [], []
    for k, v in tag_df.items():
        save_tags.append(str(k))
        dfs.append(v)

    dfs_concat = concat_dfs(dfs, drop_duplicates_columns, attach_index)

    save_dfs(dfs_concat, save_type, tag_columns, save_tags, save_path)

    return dfs_concat, save_tags


def split_large_txt_by_lines(input_file, output_dir, fmt='%d.txt', max_line=1000000, save_head=False):
    """按行将较大的文本文件划分为数个小的文本文件

    Parameters
    ----------
    input_file:
        输入文件名
    output_dir:
        输出文件夹名
    fmt:
        每个输出的小文件名所加的后缀
    max_line:
        每个小文件的最大行数
    save_head:
        每个文件是否保存第一行
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(input_file, 'r', encoding='utf8') as fin:
        head = fin.readline() if save_head else ''
        count = 0
        cache = []
        sub = 0
        for line in fin:
            if count >= max_line:
                save_path = os.path.join(output_dir, fmt % sub)
                logging.info('saving file: %s' % save_path)

                with open(save_path, 'w', encoding='utf8') as fo:
                    fo.writelines([head]) if save_head else None
                    fo.writelines(cache)

                sub += 1
                count = 0
                cache = []

            cache.append(line)
            count += 1

        if count > 0:
            save_path = os.path.join(output_dir, fmt % sub)
            logging.info('saving file: %s' % save_path)

            with open(save_path, 'w', encoding='utf8') as fo:
                fo.writelines([head]) if save_head else None
                fo.writelines(cache)


def one4all(input_file, output_file,
            line_split='\r\n', sen_split='\t', word_split=','):
    """将有若干个元素的行展开成只有两个元素的行

    Parameters
    ----------
    input_file:
        输入文件名
    output_file:
        输出文件名
    line_split:
        指定换行符
    sen_split:
        段落分隔符，默认为 '\t'
    word_split:
        单词分隔符，默认为 ','

    Examples
    --------
    >>> pd.read_table('one.txt')
      key values
    0   a  1,2,3
    >>> one4all('one.txt', 'all.txt')
    >>> pd.read_table('all.txt')
      key  values
    0   a       1
    1   a       2
    2   a       3
    """
    s = ''
    with codecs.open(input_file, 'r', encoding='utf8') as f:
        lines = f.read().split(line_split)
        for i, line in enumerate(lines):
            sentences = line.strip().split(sen_split)

            if len(sentences) == 1:  # 如果一行中只有一个元素，则采用顺序数字来代替index
                name = str(i)
                arts = sentences[0]
            else:
                name = sentences[0]
                arts = sentences[1]

            for art in arts.split(word_split):
                s += name + sen_split + art + line_split

    s = s[:-len(line_split)]
    with codecs.open(output_file, 'w', encoding='utf8') as f:
        f.write(s)


def all4one(input_file, output_file,
            line_split='\r\n', sen_split='\t', word_split=','):
    """将只有两个元素的行合并成有若干个元素的行

    Parameters
    ----------
    input_file:
        输入文件名
    output_file:
        输出文件名
    line_split:
        指定换行符
    sen_split:
        段落分隔符，默认为 '\t'
    word_split:
        单词分隔符，默认为 ','

    Examples
    --------
    >>> pd.read_table('all.txt')
      key  values
    0   a       1
    1   a       2
    2   a       3
    >>> all4one('all.txt', 'one.txt')
    >>> pd.read_table('one.txt')
      key values
    0   a  1,2,3
    """
    dic = {}
    with codecs.open(input_file, 'r', encoding='utf8') as f:
        lines = f.read().split(line_split)
        for line in lines:
            sentences = line.strip().split(sen_split)
            (name, art) = sentences
            dic[name] = dic.get(name, [])
            dic[name].append(art)
    s = ''
    for name, arts in dic.items():
        _ = ''
        for art in arts:
            _ += art + word_split
        _ = _[:-len(word_split)]

        s += name + sen_split + _ + line_split

    s = s[:-len(line_split)]
    with codecs.open(output_file, 'w', encoding='utf8') as f:
        f.write(s)
