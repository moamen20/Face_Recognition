#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os import listdir,makedirs
from os.path import join
from os.path import isdir,exists
from PIL import Image
from numpy import asarray
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from numpy import savez_compressed
import numpy as np
from keras import models  


# In[2]:

from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from matplotlib import pyplot
# load the model from disk
#filename = 'D:\GPs/finalized_model.sav'
#SVM_model= joblib.load(filename)


# ## upload a new image
# We need to :
# * detect face with mtcnn
# * save the new data to the npz file of detected faces 
# * load the data from the npz file 
# * embed the data 
# * retrain the data 
# * then done 
# 

# In[3]:


# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    #mean, std = face_pixels.mean(), face_pixels.std()
    #face_pixels = (face_pixels - mean) / std
    
    face_pixels = np.around(np.array(face_pixels) / 255.0, decimals=12)

    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    #yhat = model.predict(samples)
    yhat=model.predict_on_batch(samples)
    #return yhat[0]
    return yhat / np.linalg.norm(yhat, ord=2)


# In[4]:


# function to extract a face from an uploaded image 
def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array



def load_allfaces(directory):
    faces = list()
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        # get face
        face = extract_face(path)
        # store
        faces.append(face)
    return faces

# function to load images and extract faces for all images in a directory
def load_faces(path):
    faces = list()  
    face = extract_face(path)
    # store
    faces.append(face)
    return faces



# In[35]:


# 1st step
# detect face with mtcnn
def load_newdata(path):
    X = list()
    Y = list()
    # load all faces in the subdirectory
    faces = load_faces(path)
    # store
    X.extend(faces)  
    Y.extend(id)   
    return asarray(X), asarray(Y)

def load_dataset(directory):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
    # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_allfaces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)
 


# In[40]:


def load_newperson(directory,id):
    X, y = list(), list()
    # enumerate folders, on per class

    path = join(directory+"/",str(id)+"/") 

    # load all faces in the subdirectory
    faces = load_allfaces(path)
    # create labels
    labels = str(id)

    # store
    X.extend(faces)
    y.extend(labels)
    return asarray(X), asarray(y)


# In[45]:


def add_retrain():
    #1
   
    known_trainX, known_trainy= load_dataset('data/train/')
    print('Loaded: ', known_trainX.shape, known_trainy.shape)   
    
    #2
    
    facenet_model= models.load_model('facenet_keras.h5')
    # convert each face in the train set to an embedding  
    knownX = list()
    for face_pixels in known_trainX:
        embedding = get_embedding(facenet_model, face_pixels)
        knownX.append(embedding)
    knownX = asarray(knownX)
    print(knownX.shape)
    # save arrays to one file in compressed format
    savez_compressed('npz_files/knownfaces-embeddings.npz', knownX, known_trainy)
    
def add_newvector(id):
    facenet_model= models.load_model('facenet_keras.h5')
    #load the embedding for new knownfaces
    data = load('npz_files/knownfaces-embeddings.npz')
    x_1, y_1 = data['arr_0'],data['arr_1']
    
    #load the new face
    x_2, y_2= load_newperson('data/train/',id)
    
    knownX_new = list()
    for face_pixels in x_2:
        embedding = get_embedding(facenet_model, face_pixels)
        knownX_new.append(embedding)
    knownX_new = asarray(knownX_new)
    savez_compressed('npz_files/knownfaces-thenew_embeddings.npz', knownX_new, y_2)
    
    #load vectors for new faces 
    data_new = load('npz_files/knownfaces-thenew_embeddings.npz')
    x_2, y_2 = data_new['arr_0'],data_new['arr_1']
    
    #compine files
    x_comb=[*x_1,*x_2]
    y_comb= [*y_1,*y_2]
    savez_compressed('npz_files/knownfaces-embeddings.npz', x_comb, y_comb)
    
    
    
    
    

    


# # " **************************************************************************************** "
# # end of uploading and training 
# # start recognition 

# In[14]:


#add_retrain()


# In[15]:


def load_testdata(path):
    X = list()
    # load all faces in the subdirectory
    faces = load_faces(path)
    # store
    X.extend(faces)
    return asarray(X)


# In[23]:


def verfiy_face(input_img):
    data = load('npz_files/knownfaces-embeddings.npz')
    trainX, trainy= data['arr_0'],data['arr_1']

    loaded_X= load_testdata(input_img)
    
    facenet_model= models.load_model('facenet_keras.h5')
    encoding_test= get_embedding(facenet_model, loaded_X[0])
    mindist=5
    i=0
    similar_faces=[]
    y_face=trainy[0]
    for face in trainX:
        dist = np.linalg.norm(encoding_test - face, ord=2)
        if dist<=0.7:
            similar_faces.append(trainy[i])
        if dist<mindist:
            mindist = dist
            y_face=trainy[i]
        i+=1
    
    if mindist<0.65:
        return y_face,similar_faces
    else:
        return "unknown face"


# In[24]:

'''
import gradio as gr


demo = gr.Interface(verfiy_face, gr.Image(type="filepath"), "text")

demo.launch()


# In[29]:


def upload_img(img,id): 
    root_path="D:\GPs\data/train/"
    folderpath = join(root_path,str(id))
    makedirs(folderpath)
    img_path = join(folderpath+"/",str(id)+'.JPG')
    #img = Image.fromarray(img, 'RGB')
    img =Image.open(img)
    
    try: 
        img.save(img_path)
        add_newvector(id)
        return "added succesfully"
    except:
        return "faild"
    

    
demo_add = gr.Interface(upload_img, inputs=[gr.Image(type="filepath"),"text"], outputs=["text"])
#gr.Image(shape=(400,300))
demo_add.launch()  


'''




