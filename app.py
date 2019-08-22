from starlette.applications import Starlette
from starlette.responses import JSONResponse
import uvicorn
import requests
import os
from fastai.vision import *
from fastai.metrics import error_rate


def create_parameters():
    data_classes = [3,7]
    path = untar_data(URLs.MNIST_SAMPLE)
    if not path.ls()[2]:
        data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), size=24).normalize(imagenet_stats)
        learn = cnn_learner(data, models.resnet34, metrics=error_rate)
        learn.fit_one_cycle(4)
        learn.save('stage-1')
    return data_classes, path


class_names, stage_path = create_parameters()

app = Starlette(debug=True)

@app.route('/')
async def homepage(request):
    form = await request.form()
    if form.get('url') is not None:
        response = requests.get(form.get('url'))
        if response.status_code == 200:
            path = os.getcwd()
            file_path = os.path.join(path,"sample.jpg")
            with open(file_path, 'wb') as f:
                f.write(response.content)
                print(file_path)
                request_img = open_image(file_path)
                data2 = ImageDataBunch.single_from_classes(stage_path,classes=class_names,ds_tfms=get_transforms(),size=24).normalize(imagenet_stats)
                learn = cnn_learner(data2,models.resnet34,metrics=error_rate)
                learn.load('stage-1')
                cat,_,_ = learn.predict(request_img)
        return JSONResponse({"result": str(cat)})


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
