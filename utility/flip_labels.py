import os
import shutil

flip_label1 = "0"
flip_label2 = "1"

malicious_worker_list = ["worker0"]

data_path = "./data"
file_list = os.listdir(data_path)
file_list = [file for file in file_list if file[:6] == "worker"]

for worker in file_list:
    if worker in malicious_worker_list:
        # flip train file
        shutil.move(data_path + "/" + worker + "/train/" + flip_label1, data_path + "/" + worker + "/train/" + "temp")
        shutil.move(data_path + "/" + worker + "/train/" + flip_label2, data_path + "/" + worker + "/train/" + flip_label1)
        shutil.move(data_path + "/" + worker + "/train/" + "temp", data_path + "/" + worker + "/train/" + flip_label2)

        # flip test file
        shutil.move(data_path + "/" + worker + "/test/" + flip_label1, data_path + "/" + worker + "/test/" + "temp")
        shutil.move(data_path + "/" + worker + "/test/" + flip_label2, data_path + "/" + worker + "/test/" + flip_label1)
        shutil.move(data_path + "/" + worker + "/test/" + "temp", data_path + "/" + worker + "/test/" + flip_label2)
    else:
        print("file {0} wad pass".format(worker))
