import os
import sys
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename
import tensorflow as tf

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
			
			# Sort to show labels of first prediction in order of confidence
			top_k = predictions[0].argsort()[::-1][0:4]
			for node_id in top_k:
				human_string = label_lines[node_id]
				score = predictions[0][node_id]
				result += str('%s, \n' % (human_string))
		return result

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
