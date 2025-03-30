# StableDiffusion_t_gcode_UR5
StableDiffusion images to gcode


Intsall doker with apt

install nvidia contianer
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

check docker status
sudo systemctl status docker

use docker without sudo
sudo usermod -aG docker $USER


Construir container
docker build -t gadicc/diffusers-api .

Ejecutar container
docker run --gpus all -p 8000:8000 -e HF_AUTH_TOKEN gadicc/diffusers-api
    Descacoplado -d

Kill container
docker ps
docker stop abc123
docker kill abc123

Declarar variable de entorno 
export HF_AUTH_TOKEN="abc123"
(debe ser tu propia apikey de hugingface)

stable diffusion on docker API thks to
https://github.com/AbdBarho/stable-diffusion-webui-docker?tab=readme-ov-file

image_to_gcode, thks to #https://github.com/Stypox/image-to-gcode
#python3 image-to-gcode/image_to_gcode.py --input image.png --output graph.nc --threshold 100 --scale 1


Input example:
An scary horror house with the moon at the back at night, simple, animated


