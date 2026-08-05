# Makefile para Transformer en C y Tensor Logic

CC = gcc
CFLAGS = -O2 -Wall -lm

# Transformer tradicional
transformer: tensor.c transformer.c main.c
	$(CC) -o transformer.exe tensor.c transformer.c main.c $(CFLAGS)

# Tensor Logic
tensor_logic: tensor.c tensor_logic.c demo_tensor_logic.c
	$(CC) -o tensor_logic.exe tensor.c tensor_logic.c demo_tensor_logic.c $(CFLAGS)

# Entrenamiento con Harbour/FWH
train_harbour: tensor.c transformer.c harbour_dataset.h train_harbour.c
	$(CC) -o train_harbour.exe tensor.c transformer.c train_harbour.c $(CFLAGS)

# Todo
all: transformer tensor_logic train_harbour

# Limpiar
clean:
	del *.exe

# Ejecutar ejemplos
run_transformer: transformer
	.\transformer.exe

run_logic: tensor_logic
	.\tensor_logic.exe

run_train: train_harbour
	.\train_harbour.exe harbour_fwh_v3.jsonl 5000 20 32

.PHONY: all clean run_transformer run_logic run_train
