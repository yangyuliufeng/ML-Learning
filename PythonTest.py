# !/usr/bin/python3


def grammar(a):
    print("start learning python grammar...")

    '''
    如果改变数字数据类型的值，将重新分配内存空间
    '''
    var1 = 1
    var2 = 10
    a += var1 + var2
    del var1, var2

    list = ['Google', 'Runoob', 1997, 2000];
    list[1] = 2001
    print("list[0:2]: ", list[0:2])

    return a

grammar(5)