import numpy as np
import os
from PIL import Image
from resizeimage import resizeimage
file_name = 'img_model.npy'
training_data = []
count = 0
f = open('final.txt','r')
for x in f:
	imagepath = "images/"+str(x.partition('\t')[0])+'.jpg'
	output = str(x.partition('\t')[-1])
	output = output.replace("\n", "")
	if os.path.isfile(imagepath):
		with open(imagepath,'r+b') as f:
			print(output)
			image = Image.open(f)
			img = resizeimage.resize_cover(image, [240, 180])
			img = img.convert('RGB')
			img = np.array(img)
			training_data.append([img,output])
					
	if len(training_data) == 500:
		print(str(len(training_data))+" save point \n\n\n")
		np.save(file_name,training_data)
		training_data=[]

		
				