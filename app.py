import uvicorn
import shutil
import joblib
import os
from prediction import ReadImage
from fastapi import FastAPI, UploadFile, File
import pickle
from fastapi import Request
from model import predictImages
from VGG19 import loadmodel
# init app
app = FastAPI()


# restoring model
# pkl_filename = "model.pkl"
# with open(pkl_filename, 'rb') as file:
# autism_clf = joblib.load(file)

# routes
# Home page

@app.get("/")
def read_root():
    loadmodel()
    return f'Hello Yami'


# uploading the image
@app.post('/uploadimage')
async def iimage(image: UploadFile = File(...)):
    with open(f'{image.filename}', 'wb') as buffer:
        shutil.copyfileobj(image.file, buffer)

    return image.file


# taking the uploaded image and passing it to the model when given after unpickling the model
@app.post('/predict')
def prdict_autism(image: bytes = File(...)):
    image = ReadImage(image)
    # with open(f'{"IMAGES/" + image.filename}', 'wb') as buffer:
    #     shutil.copyfileobj(image.file, buffer)
    yami = predictImages(image.filename)
    yami.tolist()
    if yami[0][0] < yami[0][1]:
        result = "Autistic: " + str(float(str(yami[0][1])[:5]) * 100) + '%'
    else:
        result = "Normal: " + str(float(str(yami[0][0])[:5]) * 100) + '%'
    return {result}


if __name__ == '__main__':
    uvicorn.run(app, host="192.168.1.10", port=8000)
