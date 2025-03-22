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

stable diffusion on docker API thks to #https://github.com/kiri-art/docker-diffusers-api/blob/dev

image_to_gcode, thks to #https://github.com/Stypox/image-to-gcode
#python3 image-to-gcode/image_to_gcode.py --input image.png --output graph.nc --threshold 100 --scale 1

gcode viewer #https://github.com/jherrm/gcode-viewer


quiero que me ayudes a crear una interface web con gradio y python para un proyecto. La interface debe de: 1.1 Tener una entrada de escritura para ingresar un prompt (que fuera una imagen basada en el request). 1.2 Tener una entrada donde pueda cargar una imagen (o tomar una fotografía). 2 tener un botón para que automáticamente según la entrada (texto o carga de imagen) se ejecute un código. 2.1 el código que se va a ejecutar va a crear la imagen basada en el prompt o si ya se carga la imagen solo la va a procesar (DEBES BASARTE EN EL CODIGO "request-process_img.py"). 3. Tener 3 salidas de imagen (Cargada o generada, Procesada para obtener bordes, gcode generado).