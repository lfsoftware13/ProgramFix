import os

from common.pycparser_util import tokenize_by_clex_fn
import matplotlib.pyplot as plt
import numpy as np


def ge():
    for i in range(10):
        yield i


class Storage(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def draw(data_list, name_list):
    f = plt.figure(figsize=(4, 3))
    plt.plot(data_list[0][0], data_list[0][1], linewidth=2.0, label=name_list[0], linestyle="-")
    plt.plot(data_list[1][0], data_list[1][1], linewidth=2.0, label=name_list[1], linestyle="--")
    plt.plot(data_list[2][0], data_list[2][1], linewidth=2.0, label=name_list[2], linestyle=":")
    plt.plot(data_list[3][0], data_list[3][1], linewidth=2.0, label=name_list[3], linestyle="-.")
    plt.xlabel('Top-k')
    plt.ylabel('Top-k accuracy(%)')
    plt.xticks([2, 4, 6, 8, 10, 12, 14])
    plt.yticks([60, 70, 80, 90, 100])
    plt.legend(fontsize=8, )
    plt.show()
    f.savefig('score_line_figure_plot.png', bbox_inches='tight', dpi=300)

if __name__ == '__main__':

    name_list = ['N-GRAM', 'NEURAL N-GRAM', 'RNNLM', 'GRAMMAR LANGUAGE MODEL']

    data_list = [
        [(1, 58.65), (2, 70.75), (3, 76.38), (4, 79.53), (5, 81.65), (6, 83.07), (7, 84.17), (8, 85.02), (9, 85.70),
         (10, 86.28), (11, 86.77), (12, 87.19), (13, 87.57), (14, 87.90)],
        [(1, 62.79), (2, 74.50), (3, 79.88), (4, 82.85), (5, 84.84), (6, 86.25), (7, 87.34), (8, 88.17), (9, 88.85),
         (10, 89.44), (11, 89.93), (12, 90.36), (13, 90.73), (14, 91.07)],
        [(1, 65.30), (2, 76.32), (3, 81.28), (4, 84.00), (5, 85.78), (6, 87.08), (7, 88.05), (8, 88.78), (9, 89.39),
         (10, 89.90), (11, 90.33), (12, 90.72), (13, 91.05), (14, 91.35)],
        [(1, 74.24), (2, 84.54), (3, 88.70), (4, 90.85), (5, 92.30), (6, 93.23), (7, 93.90), (8, 94.40), (9, 94.80),
         (10, 95.12), (11, 95.38), (12, 95.60), (13, 95.79), (14, 95.95)]]
    data_list = [list(zip(*da)) for da in data_list]
    draw(data_list, name_list)


    # for i in ge():
    #     print(i)
    # print(os.getpid())
    # a = {'a': 1, 'b': 2}
    # s = Storage(**a)
    # print(s.a)

#     code = r'''
#
# int main ( ) { int r = 0 , g = 0 , b = 0 , i , sum = 0 , n ; scanf ( "%d" , & n ) ; char a [ 101 ] ; for ( i = 0 ; i <= n ; i ++ ) { scanf ( "%c" , & a [ i ] ) ; if ( a [ i ] == 'R' ) { r ++ ; if ( g > 1 ) sum += g - 1 ; else if ( b > 1 ) sum += b - 1 ; b = 0 ; g = 0 ; } else if ( a [ i ] == 'G' ) { g ++ ; if ( r > 1 ) sum += r - 1 ; else if ( b > 1 ) sum += b - 1 ; r = b = 0 ; } else if ( a [ i ] == 'B' ) { b ++ ; if ( g > 1 ) sum += g - 1 ; else if ( r > 1 ) sum += r - 1 ; g = r = 0 ; } } if rr r >= 1 ) sum += r - 1 ; else if ( g >= 1 ) sum += g - 1 ; else if ( b >= 1 ) sum += b - ; printf ( "%d" , sum ) ; }'''
#
#     tokenize_fn = tokenize_by_clex_fn()
#     res = tokenize_fn(code)
#     print(res)


