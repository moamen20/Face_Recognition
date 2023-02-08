
from flask import Flask, request
import json
from app_face_final import verfiy_face

app = Flask(__name__)


@app.route('/')
def hello_world():
    return "hello"


@app.route('/face_rec', methods=['GET'])
def face_recognition():
    if request.method == 'GET':
        # check if the post request has the file part
        if 'file' in request.files:
            file = request.files.get('file')
            name,similar_faces= verfiy_face(file)
            resp_data = {'name': name, 'similar faces':similar_faces}
            return json.dumps(resp_data)


if __name__ == '__main__':
    app.run(port=3000)
