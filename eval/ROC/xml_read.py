#coding: utf-8
import os
import sys
import glob
import xml.etree.ElementTree as ElementTree
from xml.dom import minidom

def xmlInfo(xmlpath, labels):
	name=[]
	xMin=[]
	yMin=[]
	xMax=[]
	yMax=[]
	xmlTree = ElementTree.parse(xmlpath)
	root = xmlTree.getroot()
	folderNode = root.find("folder")
	foldertext = 'MultipleTarget'
	pathNode = root.find("filename")
	Nodetext= pathNode.text
	sizeNode = root.find("size")
	widthNode = sizeNode.find("width")
	widthtext = widthNode.text
	heightNode = sizeNode.find("height")
	heighttext = heightNode.text
	depthNode=sizeNode.find("depth")
	depthtext=depthNode.text

	objects = []
	for objectNode in root.findall('object'):
		boxNode = objectNode.find("bndbox")
		nameNode = objectNode.find("name")
		if nameNode.text not in labels:
			continue
		xMinNode = boxNode.find("xmin")
		yMinNode = boxNode.find("ymin")
		xMaxNode = boxNode.find("xmax")
		yMaxNode = boxNode.find("ymax")
		objects.append(dict(label_name=nameNode.text, box=[xMinNode.text, yMinNode.text, xMaxNode.text, yMaxNode.text]))
	return objects 
	#toxml(doc,xmlname,foldertext,Nodetext,widthtext,heighttext,depthtext,name,xMin,yMin,xMax,yMax)

def toxml(doc,xmlname,foldertext,Nodetext,widthtext,heighttext,depthtext,name,xMin,yMin,xMax,yMax):
	root_node = doc.createElement('annotation')  
	doc.appendChild(root_node) 
  
	node = doc.createElement('folder')  
	node_text = doc.createTextNode(foldertext)  
	node.appendChild(node_text)  
	root_node.appendChild(node) 

	node = doc.createElement('filename')  
	node_text = doc.createTextNode(Nodetext)  
	node.appendChild(node_text)  
	root_node.appendChild(node)  

	source_node = doc.createElement('source')  
	root_node.appendChild(source_node) 

	node = doc.createElement('database')  
	node_text = doc.createTextNode('wissenData')  
	node.appendChild(node_text)  
	source_node.appendChild(node) 

	node = doc.createElement('annotation')  
	node_text = doc.createTextNode('wissen')  
	node.appendChild(node_text)  
	source_node.appendChild(node) 
	node = doc.createElement('image')  
	node_text = doc.createTextNode('six_class')  
	node.appendChild(node_text)  
	source_node.appendChild(node) 
     			
	owner_node = doc.createElement('owner')  
	root_node.appendChild(owner_node)

	node = doc.createElement('flickrid')  
	node_text = doc.createTextNode('NULL')  
	node.appendChild(node_text)  
	owner_node.appendChild(node)  
	node = doc.createElement('name')  
	node_text = doc.createTextNode('wissen')  
	node.appendChild(node_text)  
	owner_node.appendChild(node)  

	size_node = doc.createElement('size')  
	root_node.appendChild(size_node)

	node = doc.createElement('width')  
	node_text = doc.createTextNode(widthtext)  
	node.appendChild(node_text)  
	size_node.appendChild(node)
	node = doc.createElement('height')  
	node_text = doc.createTextNode(heighttext)  
	node.appendChild(node_text)  
	size_node.appendChild(node)
	node = doc.createElement('depth')  
	node_text = doc.createTextNode(depthtext)  
	node.appendChild(node_text)  
	size_node.appendChild(node)

	node = doc.createElement('segmented') 
	node_text = doc.createTextNode('0')  
	node.appendChild(node_text)  
	root_node.appendChild(node)

	for i in range(0,len(name)):

		object_node = doc.createElement('object')  
		root_node.appendChild(object_node)
	
		node = doc.createElement('name')  
		node_text = doc.createTextNode(str(name[i]))  
		node.appendChild(node_text)  
		object_node.appendChild(node) 

		node = doc.createElement('pose')  
		node_text = doc.createTextNode('Unspecified')  
		node.appendChild(node_text)  
		object_node.appendChild(node)  
	
		node = doc.createElement('truncated')  
		node_text = doc.createTextNode('0')  
		node.appendChild(node_text)  
		object_node.appendChild(node) 

		node = doc.createElement('difficult')  
		node_text = doc.createTextNode('0')  
		node.appendChild(node_text)  
		object_node.appendChild(node) 

		box_node = doc.createElement('bndbox')   
		object_node.appendChild(box_node) 

		node = doc.createElement('xmin')  
		node_text = doc.createTextNode(str(xMin[i]))  
		node.appendChild(node_text)  
		box_node.appendChild(node) 

		node = doc.createElement('ymin')
		node_text = doc.createTextNode(str(yMin[i]))  
		node.appendChild(node_text)  
		box_node.appendChild(node) 

		node = doc.createElement('xmax')
		node_text = doc.createTextNode(str(xMax[i]))  
		node.appendChild(node_text)  
		box_node.appendChild(node)

		node = doc.createElement('ymax')  
		node_text = doc.createTextNode(str(yMax[i]))
		node.appendChild(node_text)  
		box_node.appendChild(node)

		fw_xml = open('/home/sun/data/VOCdevkit/VOC2007/Annotations/' + xmlname, 'w')
		fw_xml.write(doc.toprettyxml(encoding='utf-8'))
		fw_xml.close()
	
'''
if __name__=='__main__':

	folderdir='/home/sun/Wissen7'
	#for name in os.listdir(folderdir):
	xmldir=folderdir+'/Annotations/'    #+name+'/Annotations1/'
	xmlPathPattern = os.path.join(str(xmldir), "*.xml")
	for xmlFilePath in glob.iglob(xmlPathPattern):
		Arr=xmlFilePath.split('/')
		xmlname=Arr[-1]
		xmlInfo(xmlFilePath,xmlname)
	print('finished')
'''
