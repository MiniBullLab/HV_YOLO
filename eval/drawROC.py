import matplotlib.pyplot as plt

def drawROC(output,labelNames):
	for labelName in labelNames:
		f=open(output+labelName+'.txt', 'r')
		results=f.readlines()

		acc = []
		rec = []
		for result in results:
			obj = result[:-1].split(">>")
			acc.append(float(obj[1].split(":")[1]))
			rec.append(float(obj[2].split(":")[1]))

		color = [0, 255, 0]
		linewidth = 0.3
		plt.plot(acc, rec, 'r*-')#, color = color, linewidth = linewidth)
		plt.plot([0.0,1.1], [0.0,1.1], 'b-')
		plt.xlim(0.0,1.1)
		plt.ylim(0.0,1.1)
		plt.title('ROC for '+ labelName +' In YOLOV3')
		plt.xlabel('ACC')
		plt.ylabel('REC')
		plt.show()

def main():
	labels = ["person"]
	drawROC("./results/", labels)

if __name__ == "__main__":
	main()

