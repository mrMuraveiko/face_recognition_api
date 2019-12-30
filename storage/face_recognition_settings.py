import face_recognition, cv2, os, numpy as np, time

def add_picture_to_dataset(dataset: list, image: np.array, filename: str):
  was_already_saved = False
  for saved_image in dataset:
    if type(saved_image[0] == image) != bool and (saved_image[0] == image).all():
      was_already_saved = True
      break

  if not was_already_saved:
    face_locations, face_encodings = get_face_embeddings_from_image(
      image, 
      convert_to_rgb=True)
  
    dataset.append([[image, filename], [face_locations, face_encodings]])
  
def make_precalculate_work_on_images(images: list, filenames: list) -> list:
  images_and_vectors = []

  for i in range(len(images)):
    add_picture_to_dataset(images_and_vectors, images[i], filenames[i])
  
  return images_and_vectors


def get_face_embeddings_from_image(image, convert_to_rgb=False):
    """
    Take a raw image and run both the face detection and face embedding model on it
    """
    # Convert from BGR to RGB if needed
    if convert_to_rgb:
        image = image[:, :, ::-1]

    # run the face detection model to find face locations
    face_locations = face_recognition.face_locations(image)

    # run the embedding model to get face embeddings for the supplied locations
    face_encodings = face_recognition.face_encodings(image, face_locations)

    return face_locations, face_encodings

def python_work(images, face_encodings_target, files, length=10):
  exclude_indexes = []
  ten_nearest_images_with_coefs = {}

  while len(ten_nearest_images_with_coefs) < length:
    i = 0
    minimum_image, filename_of_minimum_image = images[i][0]
    face_locations_minimum, face_encodings_minimum = images[i][1]
    
    while (len(face_locations_minimum) == 0 or i in exclude_indexes) and i < length:
      i += 1
      minimum_image, filename_of_minimum_image = images[i][0]
      face_locations_minimum, face_encodings_minimum = images[i][1]

    if i == len(images):
      break
    minimum_distance = face_recognition.face_distance(face_encodings_minimum, 
                                                      face_encodings_target[0])[0]
    
    exclude_index = i
    for j in range(len(images)):
      image = images[j][0][0]
      face_locations_current, face_encodings_current = images[j][1]
      
      if len(face_locations_current) == 0:
        continue
      current_distance = face_recognition.face_distance(face_encodings_current, 
                                                        face_encodings_target[0])[0]
      if current_distance < minimum_distance and j not in exclude_indexes:
        minimum_distance = current_distance
        minimum_image = image
        filename_of_minimum_image = images[j][0][1]
        exclude_index = j
    
    ten_nearest_images_with_coefs.update({filename_of_minimum_image : minimum_distance})
    exclude_indexes.append(exclude_index)

  return ten_nearest_images_with_coefs