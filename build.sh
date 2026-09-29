#!/bin/bash

CONTAINER_NAME=google-rag
IMAGE_NAME=google-rag

echo "Buildando a imagem..."
docker build -t $IMAGE_NAME .

echo "Parando container antigo (se existir)..."
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null

echo "Iniciando o container..."
docker run -d \
  -e TZ=America/Sao_Paulo \
  --name $CONTAINER_NAME \
  -p 8111:8111 \
  --restart unless-stopped \
  $IMAGE_NAME

echo "Container '$CONTAINER_NAME' rodando em http://localhost:8111"
