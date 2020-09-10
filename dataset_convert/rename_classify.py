import os
import numpy as np
from optparse import OptionParser


def parse_arguments():
    parser = OptionParser()
    parser.description = "This program rename classify image name"

    parser.add_option("-i", "--inputPath", dest="inputPath",
                      type="string", default=None,
                      help="input path of dataset")

    (options, args) = parser.parse_args()

    if options.inputPath:
        if not os.path.exists(options.inputPath):
            parser.error("Could not find the input path of dataset")
        else:
            options.input_path = os.path.normpath(options.inputPath)
    else:
        parser.error("'inputPath' option is required to run this program")

    return options

def main():
    print("process start...")
    options = parse_arguments()
    for cls_dir in os.listdir(options.inputPath):
        print("classify name: {}".format(cls_dir))
        for img_name in os.listdir(os.path.join(options.input_path, cls_dir)):
            os.rename(os.path.join(options.input_path, cls_dir, img_name),
                      os.path.join(options.input_path, cls_dir, cls_dir + "_" + img_name))

    print("End of game!!!")

if __name__ == "__main__":
    main()
