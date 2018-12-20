# -*- coding: utf-8 -*-

from tqdm import tqdm

def deal_line(line):
    '''
    spilts single line from input file to tuple of values
    '''
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


def create_submission(best, models, output_file_name, pool, weights):
    print("Weights: ", list(zip(map(lambda x: x.name, models), weights)))
    print("Best rmse: ", best[1].fun)
    print("Read submission file")

    df_submission = pd.read_csv(SAMPLE_SUBMISSION)
    df_submission = split_user_movie(df_submission)

    print("Do predictions")
    predictions = []
    items_to_predict = list(df_submission.iterrows())

    print("Split data")
    items = np.array_split(items_to_predict, 12)
    items = map(lambda x: (x, models, predictions, weights), items)

    print("Start jobs")
    p = tqdm(pool.imap(predict, items), total=12)
    new_predictions = [item for sublist in p for item in sublist]

    print("Create File")
    s.write_predictions_to_file(new_predictions, output_file_name + "_prediction.csv")


def predict(input):
    items_to_predict, models, predictions, weights = input
    print("Predict for", len(items_to_predict))

    for each in tqdm(items_to_predict):
        user = each[1].user
        movie = each[1].movie

        mix_prediction = 0
        for i, w in enumerate(models):
            pred = w.predict(user, movie)
            mix_prediction += weights[i] * pred
        predictions.append(mix_prediction)
    clipped_predictions = np.clip(predictions, 1, 5)

    new_predictions = []
    for i, each in enumerate(items_to_predict):
        one = (each[1].user, each[1].movie, clipped_predictions[i])
        new_predictions.append(one)

    return new_predictions
