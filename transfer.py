"""字符、文本替换等工具"""
import time
import re
import numpy as np

convert_dict = {0: '1', 1: '0', 2: 'X', 3: '9', 4: '8', 5: '7', 6: '6', 7: '5', 8: '4', 9: '3', 10: '2'}
param_list = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3]


def id_15_to_18(id_string: int or str) -> str:
    """15位身份证转换成18位"""

    id_string = str(id_string)

    if len(id_string) == 18:
        return id_string
    elif len(id_string) != 15:
        return ''
    if not id_string.isdigit():
        return ''

    verify_code = id_string[:6] + '19' + id_string[6:]

    num = 0
    for i in range(len(verify_code)):
        single = int(verify_code[i])
        num += single * param_list[i % 10]

    end_str = convert_dict[num % 11]

    return verify_code + end_str


def id_18_to_15(id_string: str) -> str:
    """18位身份证转换成15位"""
    
    id_string = str(id_string)

    if len(id_string) != 18:
        return ''

    return id_string[:6] + id_string[8:17]


def ip_int_to_address(ip: int or str) -> str:
    """int类型ip转ip地址段"""
    try:
        ip = int(ip)
    except:
        return ''

    if ip >= 256 ** 4:
        return ''

    address = ''
    for i in range(4):
        address = str(ip % 256) + '.' + address
        ip //=  256

    return address[:-1]


def ip_address_to_int(address: str) -> str:
    """ip地址段转int类型ip"""
    
    ip_areas = address.split('.')

    if len(ip_areas) != 4:
        return ''

    ip = 0
    for i, ip_area in enumerate(ip_areas):
        ip += int(ip_area) * (256 ** (3 - i))

    return str(ip)


def time_int_to_strftime(time_stamp: int, time_fmt='%Y-%m-%d %H:%M:%S'):
    """时间戳转标准时间格式"""

    try:
        time_stamp = int(time_stamp)
    except ValueError:
        return time_stamp

    if time_stamp < 1e4:
        return ''

    return time.strftime(time_fmt, time.localtime(time_stamp))


def time_strftime_to_int(time_str: str, time_fmt='%Y-%m-%d %H:%M:%S') -> int:
    """标准时间格式转时间戳"""
    
    return int(time.mktime(time.strptime(time_str, time_fmt)))


def rewrite(input_fn, output_fn):
    """重新写文件，剔除非utf8字符"""
    with open(input_fn, 'r', encoding='utf8', errors='ignore') as f:
        with open(output_fn, 'w', encoding='utf8') as f1:
            for line in f:
                f1.write(line)


def sub_string(string, quitwords=()):
    """去除包含关键词的行、去除@xxx、去除非中文字符"""
    if any([i in string for i in quitwords]):
        return ''
    
    string = re.sub('@.+ ', '', string)
    string = re.sub('[^\u4e00-\u9fa5]+', '', string)
    
    return string