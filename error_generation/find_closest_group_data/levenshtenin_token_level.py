


def levenshtenin_distance_iterator(leven_matrix, a_list, b_list, i, j, equal_fn=lambda a, b: a == b, max_distance=None):
    if max_distance != None and abs(i-j) > max_distance:
        leven_matrix[i][j] = max_distance + 1
        return max_distance + 1
    if leven_matrix[i][j] != None and max_distance != None and leven_matrix[i][j] > max_distance:
        return max_distance + 1

    if i == 0:
        leven_matrix[i][j] = j
        return j
    elif j == 0:
        leven_matrix[i][j] = i
        return i

    if leven_matrix[i][j] != None:
        return leven_matrix[i][j]

    bias = 0 if equal_fn(a_list[i-1], b_list[j-1]) else 1
    insert = levenshtenin_distance_iterator(leven_matrix, a_list, b_list, i, j - 1, equal_fn, max_distance) + 1
    change = levenshtenin_distance_iterator(leven_matrix, a_list, b_list, i - 1, j - 1, equal_fn, max_distance) + bias
    delete = levenshtenin_distance_iterator(leven_matrix, a_list, b_list, i - 1, j, equal_fn, max_distance) + 1

    leven_matrix[i][j] = min(insert, change, delete)

    return leven_matrix[i][j]


def levenshtenin_distance(a_list, b_list, equal_fn=lambda a, b: a == b, max_distance=None):
    a_len = len(a_list)
    b_len = len(b_list)
    matrix = make_metrix(a_len, b_len)
    try:
        res = levenshtenin_distance_iterator(matrix, a_list, b_list, a_len, b_len, equal_fn, max_distance)
    except Exception as e:
        print(e)
        return -1, matrix
    return res, matrix


def make_metrix(i, j):
    return [[None for k in range(j+1)] for o in range(i+1)]
