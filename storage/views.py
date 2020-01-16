from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.core.files.storage import FileSystemStorage

from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

from . import face_recognition_settings
import os, re, json
import numpy as np
import os
import cv2

destdir = f'{os.getcwd()}/media'


@method_decorator(csrf_exempt)
def storage(request):
    if request.method == 'POST':
        try:
            uploaded_file = request.FILES['image']
            fs = FileSystemStorage()
            if not fs.exists(uploaded_file.name):
                fs.save(uploaded_file.name, uploaded_file)
                return HttpResponse(request, status=200)
            else:
                # already exists
                return HttpResponse(request, status=409)    
        except:
            return HttpResponse(request, status=500)

    if request.method == 'GET':
        try:
            files = [f for f in next(os.walk(destdir))[2] if f.split('.')[-1] != 'npy']
            res = []
            [res.append(filename) for filename in files if re.search(r".jpg$|.jpeg$|.png$", filename)]
            return HttpResponse([res], status=200)
        except:
            return HttpResponse(status=500)

@method_decorator(csrf_exempt)
def recognize(request):
    if request.method == 'POST':
        try:                        
            files = [f for f in next(os.walk(destdir))[2] if f.split('.')[-1] != 'npy']                                  
            
            if int(json.loads(request.POST.get('content')).get('count')) <= len(files) and int(json.loads(request.POST.get('content')).get('count')) > 0:                
                res_len = int(json.loads(request.POST.get('content')).get('count'))
            else:
                if len(files) > 10:
                    res_len = 10
                else:
                    res_len = len(files)

            uploaded_file = request.FILES['image']

            tmp_file_name = "tmp." + uploaded_file.name.split('.')[-1]

            fs = FileSystemStorage()            
            fs.save(tmp_file_name, uploaded_file)
            target_image = cv2.imread(f'{destdir}/{tmp_file_name}')                                         
            
            face_locations_target, face_encodings_target = face_recognition_settings.get_face_embeddings_from_image(target_image, convert_to_rgb=True)

            fs.delete(tmp_file_name)   

            if json.loads(request.POST.get('content')).get('data'):
                data = json.loads(request.POST.get('content')).get('data')

                tmp = []
                [tmp.append(f'{filename}') for filename in data if filename in files]

                images = np.array([cv2.imread(f'{destdir}/{filename}') for filename in tmp])
                images_and_vectors = face_recognition_settings.make_precalculate_work_on_images(images, tmp)               

                if res_len > len(images):
                    res_len = len(images)
            else:
                if os.path.exists(f'{destdir}/images_and_vectors.npy'):
                    images_and_vectors = np.load(f'{destdir}/images_and_vectors.npy', allow_pickle=True)
                else:
                    images = np.array([cv2.imread(f'{destdir}/{filename}') for filename in files if re.search(r".jpg$|.jpeg$|.png$", filename)])
                    images_and_vectors = face_recognition_settings.make_precalculate_work_on_images(images, files)                                                            
            
            nearest_images_with_coefs = face_recognition_settings.python_work(images_and_vectors, face_encodings_target, files, res_len)

            tmp = []
            for key in nearest_images_with_coefs:
                tmp.append(round(nearest_images_with_coefs[key], 2))
            nearest_images_with_coefs = sorted(nearest_images_with_coefs.items(), key=lambda x: x[1])

            interval_end = max(tmp)
            amount_of_layers = 10
            step = (max(tmp) - min(tmp)) / amount_of_layers
            procents = []

            left_corner = min(tmp)
            for distance in tmp:
            if distance <= left_corner + step:
                procents.append(amount_of_layers)
            else:
                amount_of_layers -= 1
                while distance > left_corner + step:
                left_corner += step
                procents.append(amount_of_layers)
            
            result = []
            i = 0
            for key in nearest_images_with_coefs:
                result.append({"path": key[0], "result": f'{str(procents[i] * 10)}%'})
                i += 1          
                    
            return JsonResponse(res, safe=False, status=200)


def precalculate(request):
    if request.method == 'GET':
        try :
            files = [f for f in next(os.walk(destdir))[2] if f.split('.')[-1] != 'npy']

            images = np.array([cv2.imread(f'{destdir}/{filename}') for filename in files if re.search(r".jpg$|.jpeg$|.png$", filename)])
            images_and_vectors = face_recognition_settings.make_precalculate_work_on_images(images, files)

            np.save(f'{destdir}/images_and_vectors', images_and_vectors)

            return HttpResponse(status=200)
        except:
            return HttpResponse(request, status=500)
