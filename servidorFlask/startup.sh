#!/bin/bash

# Actualiza el índice de paquetes y luego instala libgl1
apt-get update && apt-get install -y libgl1 libglib2.0-0

python server.py