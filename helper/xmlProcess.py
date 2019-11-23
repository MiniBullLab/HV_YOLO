from .dataType import *
import xml.dom
import xml.dom.minidom
import xml.etree.ElementTree as ElementTree

class XMLProcess():

    MIN_WIDTH = 0
    MIN_HEIGHT = 0

    def __init__(self):
        pass

    def parseRectData(self, xmlPath):
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
        imageSize = Point2d(int(widthNode.text), int(heightNode.text))

        boxes = []
        for object_node in root.findall('rectObject'):
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
            box = Rect2D()
            box.name = name_node.text
            box.min_corner.x = xMin
            box.min_corner.y = yMin
            box.max_corner.x = xMax
            box.max_corner.y = yMax
            if box.width() >= self.MIN_WIDTH and box.height() >= self.MIN_HEIGHT:
                boxes.append(box)
        return image_name, imageSize, boxes

    def writeRectData(self, xml_path, image_name, image_size,\
                      boxes, data_set_name, folder):
        doc = xml.dom.minidom.Document()
        root = doc.createElement('annotation')
        doc.appendChild(root)

        nodeFolder = doc.createElement("folder")
        nodeFolder.appendChild(doc.createTextNode(str(folder)))
        nodeImageName = doc.createElement("filename")
        nodeImageName.appendChild(doc.createTextNode(str(image_name)))
        root.appendChild(nodeFolder)
        root.appendChild(nodeImageName)

        nodeSource = doc.createElement("source")
        nodeDatabase = doc.createElement("database")
        nodeDatabase.appendChild(doc.createTextNode(str(data_set_name)))
        nodeAnnotation = doc.createElement("annotation")
        nodeAnnotation.appendChild(doc.createTextNode("Annotations"))
        nodeSource.appendChild(nodeDatabase)
        nodeSource.appendChild(nodeAnnotation)
        root.appendChild(nodeSource)

        nodeOwner = doc.createElement("owner")
        nodeFlickrid = doc.createElement("flickrid")
        nodeFlickrid.appendChild(doc.createTextNode(str("NULL")))
        nodeName = doc.createElement("name")
        nodeName.appendChild(doc.createTextNode(str("wissen")))
        nodeOwner.appendChild(nodeFlickrid)
        nodeOwner.appendChild(nodeName)
        root.appendChild(nodeOwner)

        nodeSize = doc.createElement("size")
        nodeWidth = doc.createElement("width")
        nodeWidth.appendChild(doc.createTextNode(str(image_size.x)))
        nodeHeight = doc.createElement("height")
        nodeHeight.appendChild(doc.createTextNode(str(image_size.y)))
        nodeDepth = doc.createElement("depth")
        nodeDepth.appendChild(doc.createTextNode(str(3)))
        nodeSize.appendChild(nodeWidth)
        nodeSize.appendChild(nodeHeight)
        nodeSize.appendChild(nodeDepth)
        root.appendChild(nodeSize)

        nodeSegmented = doc.createElement("segmented")
        nodeSegmented.appendChild(doc.createTextNode(str(0)))
        root.appendChild(nodeSegmented)

        for box in boxes:
            nodeObject = doc.createElement("object")
            nodeObjectName = doc.createElement("name")
            nodeObjectName.appendChild(doc.createTextNode(str(box.name)))
            nodeObjectPose = doc.createElement("pose")
            nodeObjectPose.appendChild(doc.createTextNode(str("Unspecified")))
            nodeObjectTruncated = doc.createElement("truncated")
            nodeObjectTruncated.appendChild(doc.createTextNode(str(0)))
            nodeObjectDifficult = doc.createElement("difficult")
            nodeObjectDifficult.appendChild(doc.createTextNode(str(0)))

            nodeObjectBndbox = doc.createElement("bndbox")
            nodeObjectXmin = doc.createElement("xmin")
            nodeObjectXmin.appendChild(doc.createTextNode(str(box.min_corner.x)))
            nodeObjectYmin = doc.createElement("ymin")
            nodeObjectYmin.appendChild(doc.createTextNode(str(box.min_corner.y)))
            nodeObjectXmax = doc.createElement("xmax")
            nodeObjectXmax.appendChild(doc.createTextNode(str(box.max_corner.x)))
            nodeObjectYmax = doc.createElement("ymax")
            nodeObjectYmax.appendChild(doc.createTextNode(str(box.max_corner.y)))
            nodeObjectBndbox.appendChild(nodeObjectXmin)
            nodeObjectBndbox.appendChild(nodeObjectYmin)
            nodeObjectBndbox.appendChild(nodeObjectXmax)
            nodeObjectBndbox.appendChild(nodeObjectYmax)

            nodeObject.appendChild(nodeObjectName)
            nodeObject.appendChild(nodeObjectPose)
            nodeObject.appendChild(nodeObjectTruncated)
            nodeObject.appendChild(nodeObjectDifficult)
            nodeObject.appendChild(nodeObjectBndbox)

            root.appendChild(nodeObject)

        fp = open(xml_path, 'w')
        doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")