import os
import pandas as pd

def cache_area_code():
    """加载区域编码集

    Examples
    --------
    >>> df = cache_area_code()
    >>> df.loc['010000']
    UPAREANAME            部
    UPTYPE                2
    RELATEID        1000000
    UPCORPNAME            部
    FLAG                  0
    PROVINCEID          NaN
    PROVINCENAME        河北省
    CITYID           130500
    CITYNAME            邢台市
    TELCODE             NaN
    Name: 010000, dtype: object
    """
    fn = os.path.join(os.path.split(__file__)[0], 'data/area_code.csv')
    df = pd.read_csv(fn, sep='\t', encoding='utf8',
                     dtype={'UPAREAID': str,
                            'RELATEID': str})
    return df.set_index(keys='UPAREAID')

