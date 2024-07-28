import json
from sklearn.model_selection import train_test_split

def main(dataset, data_type):
    print("{}*******************".format(data_type))
    
    s_num = 0
    r_num = 0
    n_num = 0

    for data in dataset:
        if data['label'] == 0:
            s_num += 1
        elif data['label'] == 1:
            r_num += 1
        elif data['label'] == 2:
            n_num += 1
    
    print("SUPPORTS: ", s_num)
    print("REFUTES: ", r_num)
    print("NOT ENOUGH INFO: ", n_num)
    print("TOTAL: ", s_num + r_num + n_num)

if __name__ == '__main__':
    train_dev_path = "./data/gpt/symmetric_dev_3_all.json"
    test_path = "./data/gpt/symmetric_test_3_all.json"
    with open(train_dev_path, 'r', encoding='utf-8') as f:
        train_dev = json.load(f)
    train, dev = train_test_split(
        train_dev, test_size=0.2, random_state=42)
    with open(test_path, 'r', encoding='utf-8') as f:
        test = json.load(f)
    
    main(train, "TRAIN")
    main(dev, "DEV")
    main(test, "TEST")
