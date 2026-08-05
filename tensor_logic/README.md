# Tensor Logic - Pedro Domingos

Implementación del Tensor Logic para unificar lógica simbólica y aprendizaje numérico en una sola estructura.

## Conceptos Clave

El Tensor Logic combina:
- **Lógica difusa**: Valores de verdad en [0, 1]
- **T-normas**: Operaciones AND/OR lógicas
- **Redes de Markov Lógicas**: Reglas con pesos aprendidos
- **Inferencia probabilística**: Razonamiento bajo incertidumbre

## Operaciones Lógicas

```c
// T-norma (AND lógico)
float fuzzy_and(float a, float b, TNormType tnorm);

// T-conorma (OR lógico)  
float fuzzy_or(float a, float b);

// Complemento (NOT)
float fuzzy_not(float a);

// Implicación
float fuzzy_implies(float a, float b);
```

## T-Normas Soportadas

| T-Norma | Fórmula | Uso |
|---------|---------|-----|
| Producto | a ⊗ b = a × b | Probabilístico |
| Łukasiewicz | a ⊗ b = max(0, a+b-1) | Lógica difusa |
| Gödel | a ⊗ b = min(a, b) | Ordinal |

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `tensor_logic.h` | Estructuras y tipos |
| `tensor_logic.c` | Implementación completa |
| `demo_tensor_logic.c` | Demostraciones |

## Ejemplo: Red de Markov Lógica

```c
// Crear MLN
MarkovLogicNetwork *mln = markov_logic_create(3, 4, 2, TNORM_PRODUCT);

// Definir cláusulas
// padre(X,Y) ∧ padre(Y,Z) → abuelo(X,Z)

// Inferencia
Tensor *evidence = tensor_create((int[]){4}, 1);
evidence->data[0] = 0.9f;  // padre(Juan, María) = 0.9

Tensor *result = logical_inference(mln, evidence);
```

## Compilación

```bash
gcc -o demo_logic.exe tensor.c tensor_logic.c demo_tensor_logic.c -lm
```

## Comparación con Transformer Tradicional

| Aspecto | Transformer | Tensor Logic |
|---------|-------------|--------------|
| Atención | Score real | Score difuso |
| Entrenamiento | Backprop complejo | Pesos de cláusulas |
| Interpretabilidad | Baja | Alta |
| Datos | Muchos necesarios | Pocos suficiente |
| Razonamiento | Implícito | Explícito |
