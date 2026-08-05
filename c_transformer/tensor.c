/**
 * TENSOR.C - Implementación de Operaciones Tensoriales
 * 
 * Implementación de operaciones tensoriales para el Transformer.
 * Estas operaciones forman la base del cálculo automático
 * inspirado en la "Tensor Logic" - la unificación de
 * lógica simbólica y cálculo numérico a través de tensores.
 */

#include "tensor.h"

/* ============================================================
 * FUNCIONES DE CREACIÓN Y DESTRUCCIÓN
 * ============================================================ */

Tensor* tensor_create(int *shape, int ndim) {
    Tensor *t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) return NULL;
    
    t->ndim = ndim;
    t->shape = (int*)malloc(ndim * sizeof(int));
    if (!t->shape) { free(t); return NULL; }
    
    t->size = 1;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->size *= shape[i];
    }
    
    t->data = (float*)calloc(t->size, sizeof(float));
    if (!t->data) { free(t->shape); free(t); return NULL; }
    
    return t;
}

Tensor* tensor_create_from_data(float *data, int *shape, int ndim) {
    Tensor *t = tensor_create(shape, ndim);
    if (!t) return NULL;
    
    memcpy(t->data, data, t->size * sizeof(float));
    return t;
}

Tensor* tensor_zeros(int *shape, int ndim) {
    return tensor_create(shape, ndim);
}

Tensor* tensor_random(int *shape, int ndim) {
    Tensor *t = tensor_create(shape, ndim);
    if (!t) return NULL;
    
    // Inicialización Xavier/Glorot para mejor convergencia
    float limit = sqrt(6.0f / (float)shape[0]);
    
    for (int i = 0; i < t->size; i++) {
        t->data[i] = ((float)rand() / RAND_MAX) * 2.0f * limit - limit;
    }
    
    return t;
}

Tensor* tensor_ones(int *shape, int ndim) {
    Tensor *t = tensor_create(shape, ndim);
    if (!t) return NULL;
    
    for (int i = 0; i < t->size; i++) {
        t->data[i] = 1.0f;
    }
    
    return t;
}

void tensor_free(Tensor *t) {
    if (t) {
        if (t->data) free(t->data);
        if (t->shape) free(t->shape);
        free(t);
    }
}

Matrix* matrix_create(int rows, int cols) {
    Matrix *m = (Matrix*)malloc(sizeof(Matrix));
    if (!m) return NULL;
    
    m->rows = rows;
    m->cols = cols;
    m->data = (float*)calloc(rows * cols, sizeof(float));
    
    if (!m->data) { free(m); return NULL; }
    
    return m;
}

Matrix* matrix_create_from_data(float *data, int rows, int cols) {
    Matrix *m = matrix_create(rows, cols);
    if (!m) return NULL;
    
    memcpy(m->data, data, rows * cols * sizeof(float));
    return m;
}

void matrix_free(Matrix *m) {
    if (m) {
        if (m->data) free(m->data);
        free(m);
    }
}

/* ============================================================
 * OPERACIONES TENSORIALES BÁSICAS
 * ============================================================ */

Tensor* tensor_add(Tensor *a, Tensor *b) {
    if (a->size != b->size) {
        fprintf(stderr, "Error: Tamaños incompatibles para suma\n");
        return NULL;
    }
    
    Tensor *result = tensor_copy(a);
    if (!result) return NULL;
    
    for (int i = 0; i < result->size; i++) {
        result->data[i] += b->data[i];
    }
    
    return result;
}

Tensor* tensor_subtract(Tensor *a, Tensor *b) {
    if (a->size != b->size) {
        fprintf(stderr, "Error: Tamaños incompatibles para resta\n");
        return NULL;
    }
    
    Tensor *result = tensor_copy(a);
    if (!result) return NULL;
    
    for (int i = 0; i < result->size; i++) {
        result->data[i] -= b->data[i];
    }
    
    return result;
}

Tensor* tensor_multiply(Tensor *a, Tensor *b) {
    if (a->size != b->size) {
        fprintf(stderr, "Error: Tamaños incompatibles para multiplicación\n");
        return NULL;
    }
    
    Tensor *result = tensor_copy(a);
    if (!result) return NULL;
    
    for (int i = 0; i < result->size; i++) {
        result->data[i] *= b->data[i];
    }
    
    return result;
}

Tensor* tensor_scale(Tensor *t, float scalar) {
    Tensor *result = tensor_copy(t);
    if (!result) return NULL;
    
    for (int i = 0; i < result->size; i++) {
        result->data[i] *= scalar;
    }
    
    return result;
}

float tensor_sum(Tensor *t) {
    float sum = 0.0f;
    for (int i = 0; i < t->size; i++) {
        sum += t->data[i];
    }
    return sum;
}

float tensor_mean(Tensor *t) {
    return tensor_sum(t) / (float)t->size;
}

/* ============================================================
 * OPERACIONES MATRICIALES
 * ============================================================ */

Matrix* matrix_multiply(Matrix *a, Matrix *b) {
    if (a->cols != b->rows) {
        fprintf(stderr, "Error: Dimensiones incompatibles para multiplicación\n");
        return NULL;
    }
    
    Matrix *c = matrix_create(a->rows, b->cols);
    if (!c) return NULL;
    
    // Multiplicación de matrices estándar
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            c->data[i * b->cols + j] = sum;
        }
    }
    
    return c;
}

Matrix* matrix_multiply_transpose_a(Matrix *a, Matrix *b) {
    // C = A^T * B
    if (a->rows != b->rows) {
        fprintf(stderr, "Error: Dimensiones incompatibles\n");
        return NULL;
    }
    
    Matrix *c = matrix_create(a->cols, b->cols);
    if (!c) return NULL;
    
    for (int i = 0; i < a->cols; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a->rows; k++) {
                sum += a->data[k * a->cols + i] * b->data[k * b->cols + j];
            }
            c->data[i * b->cols + j] = sum;
        }
    }
    
    return c;
}

Matrix* matrix_multiply_transpose_b(Matrix *a, Matrix *b) {
    // C = A * B^T
    if (a->cols != b->cols) {
        fprintf(stderr, "Error: Dimensiones incompatibles\n");
        return NULL;
    }
    
    Matrix *c = matrix_create(a->rows, b->rows);
    if (!c) return NULL;
    
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->rows; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] * b->data[j * b->cols + k];
            }
            c->data[i * b->rows + j] = sum;
        }
    }
    
    return c;
}

Matrix* matrix_transpose(Matrix *m) {
    Matrix *t = matrix_create(m->cols, m->rows);
    if (!t) return NULL;
    
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            t->data[j * m->rows + i] = m->data[i * m->cols + j];
        }
    }
    
    return t;
}

Matrix* matrix_add(Matrix *a, Matrix *b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "Error: Dimensiones incompatibles\n");
        return NULL;
    }
    
    Matrix *c = matrix_create(a->rows, a->cols);
    if (!c) return NULL;
    
    for (int i = 0; i < c->rows * c->cols; i++) {
        c->data[i] = a->data[i] + b->data[i];
    }
    
    return c;
}

Matrix* matrix_subtract(Matrix *a, Matrix *b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "Error: Dimensiones incompatibles\n");
        return NULL;
    }
    
    Matrix *c = matrix_create(a->rows, a->cols);
    if (!c) return NULL;
    
    for (int i = 0; i < c->rows * c->cols; i++) {
        c->data[i] = a->data[i] - b->data[i];
    }
    
    return c;
}

Matrix* matrix_scale(Matrix *m, float scalar) {
    Matrix *c = matrix_create(m->rows, m->cols);
    if (!c) return NULL;
    
    for (int i = 0; i < c->rows * c->cols; i++) {
        c->data[i] = m->data[i] * scalar;
    }
    
    return c;
}

/* ============================================================
 * FUNCIONES DE ACTIVACIÓN
 * ============================================================ */

Tensor* tensor_relu(Tensor *t) {
    Tensor *result = tensor_copy(t);
    if (!result) return NULL;
    
    for (int i = 0; i < result->size; i++) {
        result->data[i] = fmaxf(0.0f, result->data[i]);
    }
    
    return result;
}

Tensor* tensor_gelu(Tensor *t) {
    // GELU(x) = x * Φ(x) donde Φ es la CDF de la distribución normal
    // Aproximación: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    Tensor *result = tensor_copy(t);
    if (!result) return NULL;
    
    const float sqrt_2_over_pi = sqrtf(2.0f / M_PI);
    
    for (int i = 0; i < result->size; i++) {
        float x = result->data[i];
        float inner = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
        result->data[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
    
    return result;
}

Tensor* tensor_softmax(Tensor *t) {
    Tensor *result = tensor_copy(t);
    if (!result) return NULL;
    
    // Softmax numerical stable: restamos el max
    float max_val = result->data[0];
    for (int i = 1; i < result->size; i++) {
        if (result->data[i] > max_val) {
            max_val = result->data[i];
        }
    }
    
    float sum = 0.0f;
    for (int i = 0; i < result->size; i++) {
        result->data[i] = expf(result->data[i] - max_val);
        sum += result->data[i];
    }
    
    for (int i = 0; i < result->size; i++) {
        result->data[i] /= sum;
    }
    
    return result;
}

Tensor* tensor_layer_norm(Tensor *t, Tensor *gamma, Tensor *beta) {
    // Layer Normalization: y = gamma * (x - mean) / sqrt(var + eps) + beta
    Tensor *result = tensor_copy(t);
    if (!result) return NULL;
    
    float mean = tensor_mean(t);
    
    // Calcular varianza
    float var = 0.0f;
    for (int i = 0; i < t->size; i++) {
        float diff = t->data[i] - mean;
        var += diff * diff;
    }
    var /= (float)t->size;
    
    // Normalizar
    float eps = 1e-5f;
    float std_inv = 1.0f / sqrtf(var + eps);
    
    for (int i = 0; i < result->size; i++) {
        float normalized = (t->data[i] - mean) * std_inv;
        result->data[i] = gamma->data[i % gamma->size] * normalized + 
                          beta->data[i % beta->size];
    }
    
    return result;
}

/* ============================================================
 * FUNCIONES DE UTILIDAD
 * ============================================================ */

void tensor_print(Tensor *t, const char *name) {
    if (!t) {
        printf("%s: NULL\n", name);
        return;
    }
    
    printf("%s [", name);
    for (int i = 0; i < t->ndim; i++) {
        printf("%d", t->shape[i]);
        if (i < t->ndim - 1) printf("x");
    }
    printf("]:\n");
    
    // Para tensores 1D
    if (t->ndim == 1) {
        printf("  [");
        for (int i = 0; i < t->shape[0]; i++) {
            printf("%.4f", t->data[i]);
            if (i < t->shape[0] - 1) printf(", ");
        }
        printf("]\n");
    }
    // Para tensores 2D (matrices)
    else if (t->ndim == 2) {
        for (int i = 0; i < t->shape[0]; i++) {
            printf("  [");
            for (int j = 0; j < t->shape[1]; j++) {
                printf("%.4f", t->data[i * t->shape[1] + j]);
                if (j < t->shape[1] - 1) printf(", ");
            }
            printf("]\n");
        }
    }
    // Para tensores más grandes, solo mostrar primeros elementos
    else {
        printf("  [");
        int show = (t->size > 10) ? 10 : t->size;
        for (int i = 0; i < show; i++) {
            printf("%.4f", t->data[i]);
            if (i < show - 1) printf(", ");
        }
        if (t->size > 10) printf(", ...");
        printf("]\n");
    }
}

void matrix_print(Matrix *m, const char *name) {
    if (!m) {
        printf("%s: NULL\n", name);
        return;
    }
    
    printf("%s [%dx%d]:\n", name, m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        printf("  [");
        for (int j = 0; j < m->cols; j++) {
            printf("%8.4f", m->data[i * m->cols + j]);
            if (j < m->cols - 1) printf(", ");
        }
        printf("]\n");
    }
}

Tensor* tensor_copy(Tensor *t) {
    if (!t) return NULL;
    
    Tensor *copy = tensor_create(t->shape, t->ndim);
    if (!copy) return NULL;
    
    memcpy(copy->data, t->data, t->size * sizeof(float));
    return copy;
}

Matrix* matrix_copy(Matrix *m) {
    if (!m) return NULL;
    
    Matrix *copy = matrix_create(m->rows, m->cols);
    if (!copy) return NULL;
    
    memcpy(copy->data, m->data, m->rows * m->cols * sizeof(float));
    return copy;
}

void tensor_seed(unsigned int seed) {
    srand(seed);
}
