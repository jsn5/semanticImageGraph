import os
import sys
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename
import tensorflow as tf
from nltk.corpus import wordnet as wn
from nltk.corpus import genesis
from nltk.corpus import wordnet_ic

from textblob import Word
import networkx as nx
import matplotlib.pyplot as plt


'''def wordnet_graph(w,filename):
	G=nx.Graph()

	for i in w:
		for j in w:
			if j != i:
				G.add_node(i.name())
				for h in w.hypernyms():
					  print (h)
					  G.add_node(h.name())
					  G.add_edge(w.name(),h.name())


				for h in w.hyponyms():
					  print (h)
					  G.add_node(h.name())
					  G.add_edge(w.name(),h.name())

				#print (G.nodes(data=True))
				nx.draw(G, width=2, with_labels=True)
				plt.savefig("uploads/"+filename+".png")
				plt.clf()
'''

# Initialize the Flask application
app = Flask(__name__)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'jpeg'])

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
@app.route('/')
def homepage():
	return render_template('index.html')


# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():
	result = ""
	# Get the name of the uploaded file
	file = request.files['file']
	# Check if the file is one of the allowed types/extensions
	if file and allowed_file(file.filename):
		# Make the filename safe, remove unsupported chars
		filename = secure_filename(file.filename)
		# Move the file form the temporal folder to
		# the upload folder we setup
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		# Redirect the user to the uploaded_file route, which
		# will basicaly show on the browser the uploaded file

		image_path = "uploads/"+str(filename)

		# Read in the image_data
		image_data = tf.gfile.FastGFile(image_path, 'rb').read()

		# Loads label file, strips off carriage return
		label_lines = [line.rstrip() for line 
						   in tf.gfile.GFile("retrained_labels.txt")]

		# Unpersists graph from file
		with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			tf.import_graph_def(graph_def, name='')

		with tf.Session() as sess:
			# Feed the image_data as input to the graph and get first prediction
			softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
			
			predictions = sess.run(softmax_tensor, \
					 {'DecodeJpeg/contents:0': image_data})
			
			# Sort to show labels in order of confidence
			'''top_k = predictions.argsort()[::-1][0:5]
			for node_id in top_k:
			  human_string = label_lines[node_id]
			  synset =  wn._synset_from_pos_and_offset('n',int(human_string))
			  score = predictions[node_id]
			  print('%s %s (score = %.5f)' % (human_string, synset,score))'''

		

			# Sort to show labels of first prediction in order of confidence
			top_k = predictions[0].argsort()[::-1][0:5]
			result=[]
			labels=[]
			synset=[]
			for node_id in top_k:
				human_string = label_lines[node_id]
				score = predictions[0][node_id]
				s = wn._synset_from_pos_and_offset('n',int(human_string))
				synset.append(s)
				result.append([human_string,s,score])	
			print(synset)
			wn_ic = wn.ic(wn)
			lch_list=[]
			val_map={}
			G =nx.Graph()
			for i in result:
				word_i = wn._synset_from_pos_and_offset('n',int(i[0]))
				for j in result:
					if i[0] != j[0]:
						word_j = wn._synset_from_pos_and_offset('n',int(j[0]))
						lch = word_i.lowest_common_hypernyms(word_j)

						if lch not in lch_list:
							val_map[lch[0].name()]=0.5714285714285714
							G.add_node(lch[0].name())
							lch_list.append(lch)

						if word_i.name() not in G:
							G.add_node(word_i.name())
						if word_j.name() not in G:
							G.add_node(word_j.name())
							G.add_edge(word_j.name(),lch[0].name())
						G.add_edge(word_i.name(),lch[0].name())
						
			values = [val_map.get(node, 0.15) for node in G.nodes()]
			print(lch_list)
			nx.draw(G, width=2, with_labels=True,cmap=plt.get_cmap('jet'),alpha=0.9,node_color=values,font_size=12,font_weight='bold',font_color='#ff4949')
			plt.savefig("uploads/result"+str(filename))
			plt.clf()

		return render_template('result.html',result=result,path=image_path,result_path="uploads/result"+str(filename))

# This route is expecting a parameter containing the name
# of a file. Then it will locate that file on the upload
# directory and show it on the browser, so if the user uploads
# an image, that image is going to be show after the upload
@app.route('/uploads/<filename>')
def uploaded_file(filename):
	return send_from_directory(app.config['UPLOAD_FOLDER'],
							   filename)


if __name__ == '__main__':
	app.run(debug=True, use_reloader=True)
