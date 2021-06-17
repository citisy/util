""""
usage:
    python generate_chart input_path, save_path a_index b_index
some problems:
    1. if circles in the tree, it will be meet some unknown error.
    2. output file must be .xls type
"""

import xlwt
import pandas as pd
import sys


class node:
    def __init__(self, data):
        self.data = data
        self.children = []


def main(a, b, save_path, sheet_name='Sheet1'):
    def recursion(bn, i, j):
        for child in bn.children:
            sheet.write(i, j, child.data)
            i = recursion(child, i + 1, j + 1)
        return i

    parents = {}  # {str: node}
    trees = {}  # {str: node}
    for i in range(b.shape[0]):
        p, c = b[i], a[i]
        if p == 'nan':
            continue
        if p not in parents:
            np = node(p)
            trees[p] = np
            parents[p] = np
        else:
            np = parents[p]

        if c in trees:
            nc = trees.pop(c)
        else:
            nc = node(c)
        np.children.append(nc)
        parents[c] = nc

    book = xlwt.Workbook()
    sheet = book.add_sheet(sheet_name)

    i = 0
    for root in trees.values():
        sheet.write(i, 0, root.data)
        i = recursion(root, i + 1, 1)

    book.save(save_path)


def read(input_path, a_index, b_index):
    df = pd.read_excel(input_path,
                       header=0,
                       index_col=None,
                       dtype=str,
                       na_values='')

    return df.loc[:, a_index].values, df.loc[:, b_index].values


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('param not enough! ')
    else:
        a, b = read(sys.argv[1], sys.argv[3], sys.argv[4])
        main(a, b, sys.argv[2])
