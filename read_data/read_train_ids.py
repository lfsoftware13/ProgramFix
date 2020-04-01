from config import train_ids_path, valid_ids_path, test_ids_path


def read_training_data_ids():
    train_ids = read_data_ids_from_file(train_ids_path)
    valid_ids = read_data_ids_from_file(valid_ids_path)
    test_ids = read_data_ids_from_file(test_ids_path)
    return train_ids, valid_ids, test_ids


def read_data_ids_from_file(save_path):
    with open(save_path, mode='r') as f:
        res = f.readline()
    ids = res.split(',')
    return ids


if __name__ == '__main__':
    train_ids, valid_ids, test_ids = read_training_data_ids()
    print(len(train_ids), train_ids[0])
    print(len(valid_ids), valid_ids[0])
    print(len(test_ids), test_ids[0])
    print(len(train_ids)+len(valid_ids) + len(test_ids), len(set(train_ids + valid_ids + test_ids)))
