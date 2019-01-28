
from CNN.utils import *
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

import argparse
import matplotlib.pyplot as plt
import pickle
import pandas as pd

parser = argparse.ArgumentParser(description='Predict the network accuracy.')
parser.add_argument('parameters', metavar='parameters',
                    help='name of file parameters were saved in. These parameters will be used to measure the accuracy')

if __name__ == '__main__':
    args = parser.parse_args()
    save_path = args.parameters
    
    params, cost = pickle.load(open(save_path, 'rb'))
    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    test = pd.read_csv("./Data/ntest.csv")
    test_y = np.array(test["label"])
    label_encoder = LabelEncoder()
    test_y = label_encoder.fit_transform(test_y)
    test_y = np.array(test_y.reshape(test.shape[0], 1), dtype=np.int64)
    test_X = np.array(test.drop("label", axis=1), dtype=np.float32)

    X = test_X
    y_dash = test_y

    # Normalize the Data
    # subtract mean
    X -= int(np.mean(X))
    # divide by standard deviation
    X /= int(np.std(X))
    test_data = np.hstack((X, y_dash))
    
    X = test_data[:, 0:-1]
    X = X.reshape(len(test_data), 1, 25, 25)
    y = test_data[:, -1]

    corr = 0
    digit_count = [0 for i in range(10)]
    digit_correct = [0 for i in range(10)]
   
    print()
    print("Computing accuracy over test set:")

    t = tqdm(range(len(X)), leave=True)

    for i in t:
        x = X[i]
        pred, prob = predict(x, f1, f2, w3, w4, b1, b2, b3, b4)
        digit_count[int(y[i])] += 1
        if pred == y[i]:
            corr += 1
            digit_correct[pred] += 1

        t.set_description("Acc:%0.2f%%" % (float(corr/(i+1))*100))
        
    print("Overall Accuracy: %.2f" % (float(corr/len(test_data)*100)))
    x = np.arange(10)
    digit_recall = [x/y for x, y in zip(digit_correct, digit_count)]
    plt.xlabel('Digits')
    plt.ylabel('Recall')
    plt.title("Recall on Test Set")
    plt.bar(x, digit_recall)
    plt.show()
