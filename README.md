# 🎨 StableDiffusion_t_gcode_UR5

This repository enables you to generate images using Stable Diffusion and convert them to G-code for UR5 robots. Docker is used to manage dependencies in isolated environments, making deployment easier and leveraging NVIDIA containers for GPU acceleration.

![Stable Diffusion to G-code workflow](https://via.placeholder.com/800x200)

## 📋 Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
  - [Docker Setup](#docker-setup)
  - [NVIDIA Container Toolkit](#nvidia-container-toolkit)
  - [Docker Permissions](#docker-permissions)
- [Building and Running Containers](#building-and-running-containers)
  - [Building the Container](#building-the-container)
  - [Running the Container](#running-the-container)
  - [Detached Mode](#detached-mode)
  - [Managing Containers](#managing-containers)
- [Using the Application](#using-the-application)
  - [Generating G-code from Images](#generating-g-code-from-images)
- [Example Input](#example-input)
- [Credits](#credits)

## 🖥️ Requirements

- Debian/Ubuntu-based operating system
- Docker installed
- Docker Compose
- (Optional) NVIDIA GPU with NVIDIA Container Toolkit installed

## 🔧 Installation

### Docker Setup

Update the package index and install Docker:

```bash
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```

### NVIDIA Container Toolkit

For GPU acceleration, install the NVIDIA Container Toolkit:

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Docker Permissions

To use Docker without `sudo`, add your user to the `docker` group:

```bash
sudo usermod -aG docker $USER
```

> **Note:** You may need to log out and back in for changes to take effect.

## 🐳 Building and Running Containers

### Building the Container

Download and build the image with:

```bash
docker compose --profile download up --build
```

### Running the Container

Start the container with your desired interface:

```bash
docker compose --profile [ui] up --build
```

Where `[ui]` can be one of the following profiles:
- `auto` - Automatic UI with GPU acceleration
- `auto-cpu` - Automatic UI with CPU processing
- `comfy` - ComfyUI with GPU acceleration
- `comfy-cpu` - ComfyUI with CPU processing

### Detached Mode

Run the container in the background by adding the `-d` option:

```bash
docker compose --profile [ui] up -d --build
```

### Managing Containers

List active containers:

```bash
docker ps
```

Stop a container (replace `abc123` with the appropriate ID):

```bash
docker stop abc123
```

Force stop if necessary:

```bash
docker kill abc123
```

## 🖌️ Using the Application

### Generating G-code from Images

The conversion script is based on the image-to-gcode project. Run the following command to convert an image to G-code:

```bash
python3 image-to-gcode/image_to_gcode.py --input image.png --output graph.nc --threshold 100 --scale 1
```

Make sure `image.png` is in the correct directory and adjust parameters as needed.

## Improve with
```
https://pypi.org/project/pypotrace/
```

## 🔮 Example Input

Example prompt for generating an image with Stable Diffusion:

```
An scary horror house with the moon at the back at night, simple, animated
```

## 👏 Credits

- **Stable Diffusion Docker API**: Based on [AbdBarho/stable-diffusion-webui-docker](https://github.com/AbdBarho/stable-diffusion-webui-docker)
- **image-to-gcode**: Inspired by [Stypox/image-to-gcode](https://github.com/Stypox/image-to-gcode)