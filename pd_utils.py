"""pandas数据处理常用方法"""

import pandas as pd
from tqdm import tqdm


def gb_nums(df: pd.DataFrame, gb_cols: str or list, duplicates=None) -> pd.DataFrame:
    """统计每一个分组的数量

    Parameters
    ----------
    df:
        表格数据
    gb_cols:
        需要分组的分组列名
    duplicates:
        需要去重的列名

    Returns
    -------
    df:
        每个分组数量统计的表格

    Examples
    --------
    >>> df = pd.DataFrame(['a', 'a', 'a', 'b'], columns=['gb_cols'])
    >>> df
      gb_cols
    0       a
    1       a
    2       a
    3       b
    >>> gb_nums(df, 'gb_cols')
      gb_cols  count
    0       a      3
    1       b      1
    """
    data = []
    for gb_name, df_ in tqdm(df.groupby(gb_cols)):
        if duplicates:
            df_ = df_.drop_duplicates(subset=duplicates)

        if isinstance(gb_cols, list):
            data.append(list(gb_name) + [len(df_)])
        else:
            data.append([gb_name, len(df_)])

    if isinstance(gb_cols, list):
        return pd.DataFrame(data, columns=gb_cols + ['count'])
    else:
        return pd.DataFrame(data, columns=[gb_cols, 'count'])


def gb_max(df: pd.DataFrame, gb_cols, gb_values, select_col=None) -> (pd.DataFrame, list):
    """查找每一个分组的最大值的行

    Parameters
    ----------
    df:
        表格数据
    gb_cols:
        分组列名
    gb_values:
        最大值取值的依据列名
    select_col:
        返回行数，默认返回所有最大值行

    Returns
    -------
    df, list:
        df -> 每一个分组
        list -> [分组名，最大值的行数]

    Examples
    --------
    >>> df = pd.DataFrame([('a',1),('a',2),('a',3),('b',1),('b',1)], columns=['gb_cols', 'gb_values'])
    >>> df
      gb_cols  gb_values
    0       a          1
    1       a          2
    2       a          3
    3       b          1
    4       b          1
    >>> df_, gb_list = gb_max(df, 'gb_cols', 'gb_values')
    >>> df_
      gb_cols  gb_values
    0       a          3
    1       b          1
    2       b          1
    >>> gb_list
    [(3, 1), (2, 2)]
    """
    cache = []
    gb_count = []  # (num_each_group, num_max_row)
    for _, df_ in tqdm(df.groupby(gb_cols)):
        rows = df_[df_[gb_values] == df_[gb_values].max()].values.tolist()

        if not rows:  # 找不到符合条件的行，就把原df返回
            rows = df_.values.tolist()

        if select_col:
            cache += rows[:select_col]
        else:
            cache += rows

        gb_count.append((len(df_), len(rows)))

    return pd.DataFrame(cache, columns=df.columns), gb_count


def gb_min(df: pd.DataFrame, gb_cols, gb_values, select_col=None) -> (pd.DataFrame, list):
    """查找每一个分组的最小值的行

    Parameters
    ----------
    df:
        表格数据
    gb_cols:
        分组列名
    gb_values:
        最小值取值的依据列名
    select_col:
        返回行数，默认返回所有最大值行

    Returns
    -------
    df, list:
        df -> 每一个分组，
        list -> [分组名，最小值的行数]

    Examples
    --------
    >>> df = pd.DataFrame([('a',1),('a',2),('a',3),('b',1),('b',1)], columns=['gb_cols', 'gb_values'])
    >>> df
      gb_cols  gb_values
    0       a          1
    1       a          2
    2       a          3
    3       b          1
    4       b          1
    >>> df_, gb_list = gb_min(df, 'gb_cols', 'gb_values')
    >>> df_
      gb_cols  gb_values
    0       a          1
    1       b          1
    2       b          1
    >>> gb_list
    [(3, 1), (2, 2)]
    """
    cache = []
    gb_count = []  # (num_each_group, num_min_row)
    for _, df_ in tqdm(df.groupby(gb_cols)):
        rows = df_[df_[gb_values] == df_[gb_values].min()].values.tolist()

        if not rows:  # 找不到符合条件的行，就把原df返回
            rows = df_.values.tolist()

        if select_col:
            cache += rows[:select_col]
        else:
            cache += rows

        gb_count.append((len(df_), len(rows)))

    return pd.DataFrame(cache, columns=df.columns), gb_count


def gb_same(df: pd.DataFrame, gb_cols, gb_values, select_col=None) -> (pd.DataFrame, list):
    """查找每一个分组的与分组名相同的行

    Parameters
    ----------
    df:
        表格数据
    gb_cols:
        分组列名
    gb_values:
        相同取值的依据列名
    select_col:
        返回行数，默认返回所有最大值行

    Returns
    -------
    df, list:
        df -> 每一个分组，
        list -> [分组名，相同值的行数]

    Examples
    --------
    >>> df = pd.DataFrame([('a', 'a'),('a', 'a'),('a', 'b'),('b', 'c'),('b', 'c')], columns=['gb_cols', 'gb_values'])
    >>> df
      gb_cols gb_values
    0       a         a
    1       a         a
    2       a         b
    3       b         c
    4       b         c
    >>> df_, gb_list = gb_same(df, 'gb_cols', 'gb_values')
    >>> df_
      gb_cols gb_values
    0       a         a
    1       a         a
    >>> gb_list
    [('a', 2), ('b', 0)]
    """
    cache = []
    gb_count = []  # (num_each_group, num_same_row)
    for group, df_ in tqdm(df.groupby(gb_cols)):
        rows = df_[df_[gb_values] == group].values.tolist()

        if rows:
            if select_col:
                cache += rows[:select_col]
            else:
                cache += rows

        gb_count.append((group, len(rows)))

    return pd.DataFrame(cache, columns=df.columns), gb_count


def gb_func(df: pd.DataFrame, gb_cols, gb_values, func, select_col=None) -> (pd.DataFrame, list):
    """查找每一个分组中所有都符合筛选条件的组

    Parameters
    ----------
    df:
        表格
    gb_cols:
        分组的列
    gb_values:
        值判断依据列名
    func:
        筛选函数，该函数只传入每一行 gb_values 的值
    select_col:
        返回行数，默认返回所有最大值行

    Returns
    -------
    df, list:
        df -> 每一个分组，
        list -> [分组名，符合条件行的数量]

    Examples
    --------
    >>> df = pd.DataFrame([('a', 'a'),('a', 'b'),('c', 'c'),('b', 'a'),('b', 'b')], columns=['gb_cols', 'gb_values'])
    >>> df
      gb_cols gb_values
    0       a         a
    1       a         b
    2       c         c
    3       b         a
    4       b         b
    >>> df_, gb_list = gb_func(df, 'gb_cols', 'gb_values', func=lambda x: x == 'a') # 筛选每一行 gb_values 列等于'a'的行
    >>> df_
      gb_cols gb_values
    0       a         a
    3       b         a
    >>> gb_list
    [('a', 1), ('b', 1), ('c', 0)]
    """
    cache = []
    gb_count = []  # (num_each_group, group_name)
    for group, df_ in tqdm(df.groupby(gb_cols)):
        c = []
        for _, row in df_.iterrows():
            if func(row[gb_values]):
                c.append(row)

        if c:
            if select_col:
                cache += c[:select_col]
            else:
                cache += c

        gb_count.append((group, len(c)))

    return pd.DataFrame(cache, columns=df.columns), gb_count

