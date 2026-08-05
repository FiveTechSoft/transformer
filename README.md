# Transformer Implementation

Implementación completa de Transformer en C puro con Tensor Logic de Pedro Domingos.

## Estructura

```
├── c_transformer/      # Transformer tradicional en C
│   ├── tensor.h/c      # Operaciones tensoriales
│   ├── transformer.h/c # Arquitectura completa
│   ├── training.h/c    # Sistema de entrenamiento
│   └── train_harbour.c # Entrenamiento con datasets
│
└── tensor_logic/       # Tensor Logic (Domingos)
    ├── tensor_logic.h/c # Lógica difusa + MLN
    └── demo_tensor_logic.c # Demostraciones
```

## Compilación Rápida

```bash
# Transformer tradicional
gcc -o transformer.exe c_transformer/tensor.c c_transformer/transformer.c c_transformer/main.c -lm

# Tensor Logic
gcc -o tensor_logic.exe c_transformer/tensor.c tensor_logic/tensor_logic.c tensor_logic/demo_tensor_logic.c -lm

# Con CUDA (GPU)
nvcc -o train_cuda.exe train_cuda.cu
```

## Características

### Transformer Tradicional
- Multi-Head Self-Attention
- Position-wise Feed-Forward (GELU)
- Positional Encoding sinusoidal
- Layer Normalization
- Encoder-Decoder completo

### Tensor Logic
- Lógica difusa (fuzzy logic)
- T-normas: Producto, Łukasiewicz, Gödel
- Redes de Markov Lógicas
- Atención lógica
- Inferencia probabilística

## GPU Support

Entrenamiento acelerado con CUDA para GPUs NVIDIA:
- RTX 3060/3070/3080
- RTX 4060/4070/4080/4090
- Tesla series

## Referencias

- Vaswani et al. "Attention Is All You Need" (2017)
- Domingos, P. "The Master Algorithm" (2015)
- Markov Logic Networks

## License

MIT
