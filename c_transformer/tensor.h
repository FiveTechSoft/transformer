/**
 * TENSOR.H - Operaciones Tensoriales para Transformer en C
 * 
 * Implementación de operaciones tensoriales básicas inspiradas
 * en los principios de diferenciación automática y cálculo tensorial.
 * 
 * Basado en conceptos de Pedro Domingos sobre la unificación
 * de lógica y aprendizaje a través de tensores.
 */

#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

/* ============================================================
 * ESTRUCTURAS DE DATOS TENSORIALES
 * ============================================================ */

/**
 * Tensor - Estructura básica para representar tensores
 * Un tensor es una generalización de vectores y matrices
 */
typedef struct {
    float *data;        // Datos del tensor
    int *shape;         // Forma del tensor [d0, d1, ..., dn]
    int ndim;           // Número de dimensiones
    int size;           // Tamaño total de elementos
} Tensor;

/**
 * Matriz - Representación simplificada para operaciones 2D
 */
typedef struct {
    float *data;
    int rows;
    int cols;
} Matrix;

/* ============================================================
 * FUNCIONES DE CREACIÓN Y DESTRUCCIÓN
 * ============================================================ */

// Crear tensor con forma específica
Tensor* tensor_create(int *shape, int ndim);

// Crear tensor con datos inicializados
Tensor* tensor_create_from_data(float *data, int *shape, int ndim);

// Crear tensor de zeros
Tensor* tensor_zeros(int *shape, int ndim);

// Crear tensor con valores aleatorios (Xavier/Glorot)
Tensor* tensor_random(int *shape, int ndim);

// Crear tensor con una dimensión constante
Tensor* tensor_ones(int *shape, int ndim);

// Liberar memoria del tensor
void tensor_free(Tensor *t);

// Crear matriz
Matrix* matrix_create(int rows, int cols);

// Crear matriz con datos
Matrix* matrix_create_from_data(float *data, int rows, int cols);

// Liberar matriz
void matrix_free(Matrix *m);

/* ============================================================
 * OPERACIONES TENSORIALES BÁSICAS
 * ============================================================ */

// Suma de tensores (broadcasting)
Tensor* tensor_add(Tensor *a, Tensor *b);

// Resta de tensores
Tensor* tensor_subtract(Tensor *a, Tensor *b);

// Multiplicación elemento a elemento
Tensor* tensor_multiply(Tensor *a, Tensor *b);

// Multiplicación escalar
Tensor* tensor_scale(Tensor *t, float scalar);

// Suma de todos los elementos
float tensor_sum(Tensor *t);

// Promedio de todos los elementos
float tensor_mean(Tensor *t);

/* ============================================================
 * OPERACIONES MATRICIALES
 * ============================================================ */

// Multiplicación de matrices: C = A * B
Matrix* matrix_multiply(Matrix *a, Matrix *b);

// Multiplicación transpuesta: C = A^T * B
Matrix* matrix_multiply_transpose_a(Matrix *a, Matrix *b);

// Multiplicación transpuesta: C = A * B^T
Matrix* matrix_multiply_transpose_b(Matrix *a, Matrix *b);

// Transponer matriz
Matrix* matrix_transpose(Matrix *m);

// Suma de matrices
Matrix* matrix_add(Matrix *a, Matrix *b);

// Resta de matrices
Matrix* matrix_subtract(Matrix *a, Matrix *b);

// Escalar matriz
Matrix* matrix_scale(Matrix *m, float scalar);

/* ============================================================
 * FUNCIONES DE ACTIVACIÓN
 * ============================================================ */

// ReLU
Tensor* tensor_relu(Tensor *t);

// GELU (Gaussian Error Linear Unit) - Usado en Transformers
Tensor* tensor_gelu(Tensor *t);

// Softmax
Tensor* tensor_softmax(Tensor *t);

// Layer Normalization
Tensor* tensor_layer_norm(Tensor *t, Tensor *gamma, Tensor *beta);

/* ============================================================
 * FUNCIONES DE UTILIDAD
 * ============================================================ */

// Imprimir tensor
void tensor_print(Tensor *t, const char *name);

// Imprimir matriz
void matrix_print(Matrix *m, const char *name);

// Copiar tensor
Tensor* tensor_copy(Tensor *t);

// Copiar matriz
Matrix* matrix_copy(Matrix *m);

// Inicializar semilla aleatoria
void tensor_seed(unsigned int seed);

#endif /* TENSOR_H */
