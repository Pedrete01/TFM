#!/bin/bash

# Instalar dependencias del sistema
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Ejecutar el comando original para iniciar la aplicación
python server.py
