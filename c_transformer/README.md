# Transformer en C

Implementación completa de un Transformer en lenguaje C puro, siguiendo el paper "Attention Is All You Need".

## Características

- **Multi-Head Self-Attention**: Mecanismo de atención multi-cabeza
- **Position-wise Feed-Forward**: Redes feed-forward posición por posición  
- **Positional Encoding**: Codificación posicional sinusoidal
- **Layer Normalization**: Normalización de capas
- **Encoder-Decoder**: Arquitectura completa

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `tensor.h/c` | Operaciones tensoriales básicas |
| `transformer.h/c` | Arquitectura Transformer completa |
| `training.h/c` | Sistema de entrenamiento |
| `harbour_dataset.h` | Parser para datasets JSONL |
| `train_harbour.c` | Entrenamiento con Harbour/FWH dataset |
| `main.c` | Ejemplos de uso |

## Compilación

```bash
gcc -o transformer.exe tensor.c transformer.c main.c -lm
```

## Uso

```bash
# Ejemplo básico
./transformer.exe

# Entrenar con dataset
./train_harbour.exe harbour_fwh_v3.jsonl 5000 20 32
```

## Parámetros por defecto

- `vocab_size`: 1000
- `d_model`: 512  
- `n_heads`: 8
- `d_ff`: 2048
- `n_layers`: 6
- `max_seq_len`: 512
