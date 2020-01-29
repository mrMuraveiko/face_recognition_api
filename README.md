# Face Recognition API


git clone

docker build -t {docker image name} -f Dockerfile .

docker run -it -p 8888:8888 {docker image name}

## Endpoints

| Method    | URL                | Body                                  |
| :-------: | :----------------: | :-----------------------------------: |
| POST      | api/v1/storage/    | image: unique name.[jpg, jpeg, png]   | 
| POST      | api/v1/recognize/  | image: name.[jpg, jpeg, png]          |
