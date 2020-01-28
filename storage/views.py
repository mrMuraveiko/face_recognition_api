from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.core.files.storage import FileSystemStorage

from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

from . import face_recognition_settings
import os
import re
import json
import numpy as np
import os
import cv2

destdir = f'{os.getcwd()}/media'


@method_decorator(csrf_exempt)
def storage(request):
    if request.method == 'POST':
        try:
            if os.path.exists(f'{destdir}/images_and_vectors.npy'):
                images_and_vectors = np.load(
                    f'{destdir}/images_and_vectors.npy', allow_pickle=True)
            else:
                images_and_vectors = []

            uploaded_file = request.FILES['image']
            extension = uploaded_file.name.split('.')[-1]
            if extension != 'jpg' or extension != 'jpeg' or extension != 'png':
                return HttpResponse(request, status=409)

            tmp_file_name = f"tmp_{uploaded_file.name}." + extension

            fs = FileSystemStorage()
            fs.save(tmp_file_name, uploaded_file)
            target_image = cv2.imread(f'{destdir}/{tmp_file_name}')

            fs.delete(tmp_file_name)

            if face_recognition_settings.add_picture_to_dataset(images_and_vectors, uploaded_file.name, cv2.imread(target_image)):
                np.save(f'{destdir}/images_and_vectors', images_and_vectors)

                return HttpResponse(request, status=200)
            else:
                return HttpResponse(request, status=409)
        except:
            return HttpResponse(request, status=500)


@method_decorator(csrf_exempt)
def recognize(request):
    if request.method == 'POST':
        try:
            if !os.path.exists(f'{destdir}/images_and_vectors.npy'):
                return HttpResponse(request, status=409)

            images_and_vectors = np.load(
                f'{destdir}/images_and_vectors.npy', allow_pickle=True)

            uploaded_file = request.FILES['image']
            extension = uploaded_file.name.split('.')[-1]
            if extension != 'jpg' or extension != 'jpeg' or extension != 'png':
                return HttpResponse(request, status=409)

            tmp_file_name = f"tmp_{uploaded_file.name}." + extension

            fs = FileSystemStorage()
            fs.save(tmp_file_name, uploaded_file)
            target_image = cv2.imread(f'{destdir}/{tmp_file_name}')

            face_locations_target, face_encodings_target = face_recognition_settings.get_face_embeddings_from_image(
                target_image, convert_to_rgb=True)

            fs.delete(tmp_file_name)

            nearest_images_with_coefs = face_recognition_settings.python_work(
                images_and_vectors, face_encodings_target, len(images_and_vectors))

            tmp = []
            for key in nearest_images_with_coefs:
                tmp.append(round(nearest_images_with_coefs[key], 2))
            nearest_images_with_coefs = sorted(
                nearest_images_with_coefs.items(), key=lambda x: x[1])

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
                result.append(
                    {"path": key[0], "result": f'{str(procents[i] * 10)}%'})
                i += 1

            return JsonResponse(res, safe=False, status=200)
