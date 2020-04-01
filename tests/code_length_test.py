import json


if __name__ == '__main__':
    with open('./text_file/cparser_code_length.txt') as f:
        s = f.read()
    c_d = json.loads(s)

    with open('./text_file/deepfix_code_length.txt') as f:
        s = f.read()
    d_d = json.loads(s)

    count = 0
    for k in c_d.keys():
        if c_d[k] != d_d[k]:
            print(k, c_d[k], d_d[k])
            count += 1
    print(count)