import numpy as np
import pandas as pd

def data_frame_to_matrix(file_path = 'data_train.csv'):
    data = pd.read_csv(file_path)

    # n=2 limits the output to two items
    id_splited = data['Id'].str.split('_', n=2, expand=True)
    data['User'] = id_splited[0].str.extract('(\d+)', expand=True)
    data['Movie'] = id_splited[1].str.extract('(\d+)', expand=True)

    M = np.zeros((10000, 1000))

    length = len(data)
    for i in range(0, length):
        user_id = int(data.iloc[i, 2])
        movie_id = int(data.iloc[i, 3])
        rating = data.iloc[i, 1]
        M[user_id - 1, movie_id - 1] = rating
    return M


def matrix_to_data_frame(M):
    n_rows = len(M)
    n_columns = len(M[0])
    non_zero = np.count_nonzero(M)
    sep = '_'
    cells = ["" for x in range(non_zero)]
    predictions = ["" for x in range(non_zero)]
    counter = 0
    for i in range(0, n_columns):
        for j in range(0, n_rows):
            if M[j, i] != 0:
                cells[counter] = 'r' + str(j + 1) + sep + 'c' + str(i + 1)
                predictions[counter] = M[j, i]
                counter = counter + 1

    d = {'Id': cells, 'Prediction': counter}
    df = pd.DataFrame(data=d)
    return df


# I have to convert the
def deal_line(line):
    pos, rating = line.split(',')
    row, col = pos.split("_")
    row = row.replace("r", "")
    col = col.replace("c", "")
    return int(row), int(col), float(rating)

def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()

def read_data(file_path = 'data/data_train.csv'):
    lines = read_txt(file_path)[1:]
    data = [deal_line(line) for line in lines]
    return data


def convert_to_data(file_path = 'data/data_train.csv',
                    sub_sample = False):
    data = read_data(file_path)

    if sub_sample:
        shuffle(data)
        data = data[:100_000]

    with open('data/kiru.csv', 'w') as f:
        for item in data:
            f.write("{},{},{}\n".format(item[0], item[1], item[2]))

    # path to dataset file
    file_path = 'data/kiru.csv'
    reader = Reader(line_format='user item rating', sep=',')
    data = Dataset.load_from_file(file_path, reader=reader)
    return data


