# -*- coding: utf-8 -*-

from tqdm import tqdm

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



def create_predictions(model ):
    lines = read_txt('data/sample_submission.csv')[1:]
    data = [deal_line(line) for line in lines]

    predictions = []
    for each in tqdm(data):
        pred1 = model.predict(str(each[0]), str(each[1])).est
        predictions.append((each[0], each[1], pred1))

    return predictions

def write_predictions_to_file(predictions, output_file):
    with open(output_file, 'w') as f:
        f.write("Id,Prediction\n")
        for item in tqdm(predictions):
            f.write("r{}_c{},{}\n".format(item[0], item[1], int(round(item[2]))))
    return predictions

def create_submission_file(model, output_file):
    lines = read_txt('data/sample_submission.csv')[1:]
    data = [deal_line(line) for line in lines]

    predictions = []
    for each in tqdm(data):
        pred1 = model.predict(str(each[0]), str(each[1])).est
        predictions.append((each[0], each[1], pred1))

    write_predictions_to_file(predictions, output_file)
    return predictions
