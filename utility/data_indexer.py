import os

DATA_PATH = "./data"

label_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
# label_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'horse', 'frog', 'ship', 'truck']

file_list = os.listdir(DATA_PATH)
file_filter_list = [file for file in file_list if file[0:6] == "worker"]

worker_train_index_dict = {}
worker_test_index_dict = {}

for worker in file_filter_list:
    train = os.listdir(DATA_PATH + "/" + worker + "/train/")
    train_filter = [label for label in train if label in label_list]

    test = os.listdir(DATA_PATH + "/" + worker + "/test/")
    test_filter = [label for label in test if label in label_list]

    print("{0} data indexing!".format(worker))

    train_label_dict = {}
    test_label_dict = {}

    for label in train_filter:
        train_index = os.listdir(DATA_PATH + "/" + worker + "/train/" + label)
        train_index_filter = [index for index in train_index if index.endswith(".png") or index.endswith(".jpg") or index.endswith(".jpeg")]
        # train_label_dict = dict.fromkeys(label, train_index_filter)
        train_label_dict[str(label)] = train_index_filter
        if worker in worker_train_index_dict:
            worker_train_index_dict[worker].update(train_label_dict)
        else:
            worker_train_index_dict[worker] = train_label_dict

    for label in test_filter:
        test_index = os.listdir(DATA_PATH + "/" + worker + "/test/" + label)
        test_index_filter = [index for index in test_index if index.endswith(".png") or index.endswith(".jpg") or index.endswith(".jpeg")]
        # test_label_dict = dict.fromkeys(label, test_index_filter)
        test_label_dict[str(label)] = test_index_filter
        if worker in worker_test_index_dict:
            worker_test_index_dict[worker].update(test_label_dict)
        else:
            worker_test_index_dict[worker] = test_label_dict


train_index_file = open("./train_index", "w")
train_index_file.write(str(worker_train_index_dict))
train_index_file.close()
print(worker_train_index_dict)

test_index_file = open("./test_index", "w")
test_index_file.write(str(worker_test_index_dict))
test_index_file.close()
print(worker_test_index_dict)