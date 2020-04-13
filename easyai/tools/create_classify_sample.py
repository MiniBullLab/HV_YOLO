import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

import codecs
import json
import random
import numpy as np
from easyai.helper.dirProcess import DirProcess
import cv2


class CreateClassifySample():

    def __init__(self):
        self.dir_process = DirProcess()

    def process_sample(self, input_dir, output_dir, flag, probability=1):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if "train_val" == flag.strip():
            self.process_train_val(input_dir, output_dir, probability)
        elif "train" == flag.strip():
            self.process_train(input_dir, output_dir, flag)
        elif "val" == flag.strip():
            self.process_train(input_dir, output_dir, flag)

    def process_train_val(self, input_dir, output_dir, probability):
        data_class = self.get_data_class(input_dir)

        save_train_path = os.path.join(output_dir, "train.txt")
        save_val_path = os.path.join(output_dir, "val.txt")
        save_train_file = open(save_train_path, "w")
        save_val_file = open(save_val_path, "w")

        for class_index, class_name in enumerate(data_class):
            data_class_dir = os.path.join(input_dir, class_name)
            image_list = list(self.dir_process.getDirFiles(data_class_dir,
                                                           "*.*"))
            random.shuffle(image_list)
            for image_index, image_path in enumerate(image_list):
                print(image_path)
                if (image_index + 1) % probability == 0:
                    self.write_data(image_path, class_name, class_index, save_val_file)
                else:
                    self.write_data(image_path, class_name, class_index, save_train_file)

        save_train_file.close()
        save_val_file.close()
        self.write_data_class(data_class, output_dir)

    def process_train(self, input_dir, output_dir, flag):
        data_class = self.get_data_class(input_dir)

        save_train_path = os.path.join(output_dir, "%s.txt" % flag)
        save_train_file = open(save_train_path, "w")

        for class_index, class_name in enumerate(data_class):
            data_class_dir = os.path.join(input_dir, class_name)
            image_list = list(self.dir_process.getDirFiles(data_class_dir,
                                                           "*.*"))
            random.shuffle(image_list)
            for image_index, image_path in enumerate(image_list):
                print(image_path)
                self.write_data(image_path, class_name, class_index, save_train_file)

        save_train_file.close()
        self.write_data_class(data_class, output_dir)

    def write_data(self, image_path, class_name, class_index, save_file):
        path, fileNameAndPost = os.path.split(image_path)
        fileName, post = os.path.splitext(fileNameAndPost)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
        if image is not None:
            write_content = "%s/%s %d\n" % (class_name, fileNameAndPost,
                                            class_index)
            save_file.write(write_content)

    def get_data_class(self, data_dir):
        result = []
        dir_names = os.listdir(data_dir)
        for name in dir_names:
            if not name.startswith("."):
                file_path = os.path.join(data_dir, name)
                if os.path.isdir(file_path):
                    result.append(name)
        return sorted(result)

    def write_data_class(self, data_class, output_dir):
        class_define = {}
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for index, className in enumerate(data_class):
            class_define[index] = className
        save_class_path = os.path.join(output_dir, "class.json")
        with codecs.open(save_class_path, 'w', encoding='utf-8') as f:
            json.dump(class_define, f, sort_keys=True, indent=4, ensure_ascii=False)


def main():
    print("start...")
    test = CreateClassifySample()
    test.process_sample("/home/lpj/github/data/cifar100/JPEGImages",
                        "/home/lpj/github/data/cifar100/ImageSets",
                        "train_val",
                        10)
    print("End of game, have a nice day!")


if __name__ == "__main__":
    main()

