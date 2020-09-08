import os
import cv2
import json
import numpy as np
import xml.dom
import xml.dom.minidom
import xml.etree.ElementTree as ElementTree
from optparse import OptionParser


def parse_arguments():
    parser = OptionParser()
    parser.description = "This program transform xml to json"

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


def parseRectData(xmlPath):
    xmlTree = ElementTree.parse(xmlPath)
    root = xmlTree.getroot()
    folder_node = root.find("folder")
    folder = folder_node.text
    image_node = root.find("filename")
    image_name = image_node.text
    # print(image_name)
    imageSizeNode = root.find("size")
    widthNode = imageSizeNode.find("width")
    heightNode = imageSizeNode.find("height")
    depthNode = imageSizeNode.find("depth")
    image_size = [int(widthNode.text), int(heightNode.text), int(depthNode.text)]

    num = 0
    boxes = []
    for object_node in root.findall('rectObject'):
        num += 1
        box_node = object_node.find("bndbox")
        name_node = object_node.find("name")
        xMinNode = box_node.find("xmin")
        yMinNode = box_node.find("ymin")
        xMaxNode = box_node.find("xmax")
        yMaxNode = box_node.find("ymax")
        xMin = int(xMinNode.text)
        yMin = int(yMinNode.text)
        xMax = int(xMaxNode.text)
        yMax = int(yMaxNode.text)
        box = []
        box.append(name_node.text)
        box.append(xMin)
        box.append(yMin)
        box.append(xMax)
        box.append(yMax)
        boxes.append(box)
    return image_name, image_size, boxes, num


def json_write(database, file_name, file_path, image_size, num, boxes):
    annotation = {}
    # annotation
    annotation['annotation'] = 'Annotations'
    # database
    annotation['database'] = database
    # owner
    annotation['owner'] = 'miniBull'
    # folder
    annotation['folder'] = 'JPEGImages'
    # filename
    annotation['filename'] = file_name
    # path
    annotation['path'] = file_path
    # size
    annotation['size'] = {'width': image_size[0], 'height': image_size[1], 'depth': image_size[2]}
    # objectCount
    annotation['objectCount'] = num
    # objects
    rectObject = []
    for box in boxes:
        rectObject.append({'class': box[0], 'minX': box[1], 'minY': box[2], 'maxX': box[3], 'maxY': box[4]})
    annotation['objects'] = {'rectObject': rectObject}

    json_path = file_path.replace("JPEGImages", "Annotations_json").replace("jpg", "json").replace("png", "json")
    a = json.dumps(annotation, indent=4)
    f = open(json_path, 'w')
    f.write(a)
    f.close()


def main():
    print("process start...")
    options = parse_arguments()
    if not os.path.exists(options.inputPath.replace("JPEGImages", "Annotations_json")):
        os.mkdir(options.inputPath.replace("JPEGImages", "Annotations_json"))
    for img_name in os.listdir(options.inputPath):
        img_path = os.path.join(options.inputPath, img_name)
        xml_path = img_path.replace("JPEGImages", "Annotations").replace(".jpg", ".xml").replace(".png", ".xml")
        print(xml_path)
        if not os.path.exists(xml_path):
            continue
        image_name, image_size, boxes, num = parseRectData(xml_path)
        json_write("DataSet", image_name, img_path, image_size, num, boxes)


if __name__ == "__main__":
    main()
