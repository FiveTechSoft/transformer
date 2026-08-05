/**
 * TRAINING.H - Sistema de Entrenamiento para Transformer
 * 
 * Implementa backpropagation, función de pérdida y optimizador Adam
 * para entrenar el Transformer con datos reales.
 */

#ifndef TRAINING_H
#define TRAINING_H

#include "transformer.h"

/* ============================================================
 * FUNCIONES DE PÉRDIDA
 * ============================================================ */

// Cross-Entropy Loss (para clasificación de tokens)
float cross_entropy_loss(Tensor *logits, Tensor *targets);

// Gradient de Cross-Entropy con Softmax
Tensor* cross_entropy_backward(Tensor *logits, Tensor *targets);

/* ============================================================
 * OPTIMIZADOR ADAM
 * ============================================================ */

typedef struct {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    int timestep;
    
    // Momentos para cada parámetro
    Tensor *m;  // Primer momento (media)
    Tensor *v;  // Segundo momento (varianza)
} AdamOptimizer;

// Crear optimizador Adam
AdamOptimizer* adam_create(float lr, float beta1, float beta2, float eps, float wd);

// Actualizar pesos con Adam
void adam_update(AdamOptimizer *opt, Tensor *params, Tensor *grads);

// Liberar optimizador
void adam_free(AdamOptimizer *opt);

/* ============================================================
 * BACKPROPAGATION
 * ============================================================ */

// Gradientes del Transformer
typedef struct {
    // Embedding gradients
    Matrix *enc_emb_grad;
    Matrix *dec_emb_grad;
    
    // Encoder layer gradients
    Tensor **enc_ln1_gamma_grad;
    Tensor **enc_ln1_beta_grad;
    Tensor **enc_ln2_gamma_grad;
    Tensor **enc_ln2_beta_grad;
    
    // Decoder layer gradients
    Tensor **dec_ln1_gamma_grad;
    Tensor **dec_ln1_beta_grad;
    Tensor **dec_ln2_gamma_grad;
    Tensor **dec_ln2_beta_grad;
    Tensor **dec_ln3_gamma_grad;
    Tensor **dec_ln3_beta_grad;
    
    // Attention gradients (simplificado)
    Tensor **enc_attn_grads;
    Tensor **dec_attn_grads;
    
    // FFN gradients (simplificado)
    Tensor **enc_ffn_grads;
    Tensor **dec_ffn_grads;
    
    int n_layers;
} TransformerGradients;

// Calcular gradientes (backward pass simplificado)
TransformerGradients* transformer_backward(Transformer *model, 
                                           Tensor *logits, 
                                           Tensor *targets);

// Aplicar gradientes
void transformer_apply_grads(Transformer *model, 
                             TransformerGradients *grads,
                             AdamOptimizer *opt);

// Liberar gradientes
void gradients_free(TransformerGradients *grads);

/* ============================================================
 * DATOS DE ENTRENAMIENTO
 * ============================================================ */

typedef struct {
    Tensor **encoder_inputs;
    Tensor **decoder_inputs;
    Tensor **targets;
    int n_samples;
    int seq_len;
    int vocab_size;
} TrainingBatch;

// Crear batch de entrenamiento
TrainingBatch* batch_create(int n_samples, int seq_len, int vocab_size);

// Generar datos sintéticos para demostración
TrainingBatch* generate_synthetic_data(int n_samples, int seq_len, int vocab_size);

// Liberar batch
void batch_free(TrainingBatch *batch);

/* ============================================================
 * BUCLE DE ENTRENAMIENTO
 * ============================================================ */

typedef struct {
    float train_loss;
    float learning_rate;
    int epoch;
    int step;
} TrainingStats;

// Entrenar por un epoch
TrainingStats train_epoch(Transformer *model, 
                          TrainingBatch *batch,
                          AdamOptimizer *opt,
                          int batch_size);

// Entrenar modelo completo
void train_transformer(Transformer *model, 
                       int n_epochs,
                       int n_samples,
                       int seq_len,
                       int batch_size,
                       float learning_rate);

#endif /* TRAINING_H */
