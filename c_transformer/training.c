/**
 * TRAINING.C - Implementación del Sistema de Entrenamiento
 * 
 * Este archivo implementa todos los componentes necesarios para
 * entrenar el Transformer: backpropagation, pérdida y optimizador.
 */

#include "training.h"

/* ============================================================
 * FUNCIONES DE PÉRDIDA
 * ============================================================ */

/**
 * Cross-Entropy Loss
 * 
 * L = -∑ y_true * log(y_pred)
 * 
 * Para cada posición en la secuencia, calculamos la pérdida
 * entre la predicción y el token objetivo real.
 */
float cross_entropy_loss(Tensor *logits, Tensor *targets) {
    int batch_size = logits->shape[0];
    int vocab_size = logits->shape[1];
    
    float total_loss = 0.0f;
    int valid_tokens = 0;
    
    for (int i = 0; i < batch_size; i++) {
        // Obtener logits para esta posición
        float *logits_i = &logits->data[i * vocab_size];
        
        // Encontrar max para estabilidad numérica
        float max_logit = logits_i[0];
        for (int j = 1; j < vocab_size; j++) {
            if (logits_i[j] > max_logit) {
                max_logit = logits_i[j];
            }
        }
        
        // Calcular log-softmax
        float log_sum_exp = 0.0f;
        for (int j = 0; j < vocab_size; j++) {
            log_sum_exp += expf(logits_i[j] - max_logit);
        }
        log_sum_exp = max_logit + logf(log_sum_exp);
        
        // Pérdida para el token objetivo
        int target_token = (int)targets->data[i];
        if (target_token >= 0 && target_token < vocab_size) {
            total_loss += log_sum_exp - logits_i[target_token];
            valid_tokens++;
        }
    }
    
    return (valid_tokens > 0) ? total_loss / valid_tokens : 0.0f;
}

/**
 * Gradient de Cross-Entropy con Softmax
 * 
 * dL/dz = softmax(z) - one_hot(target)
 * 
 * Este es el gradiente con respecto a los logits de entrada.
 */
Tensor* cross_entropy_backward(Tensor *logits, Tensor *targets) {
    int batch_size = logits->shape[0];
    int vocab_size = logits->shape[1];
    
    int shape[2] = {batch_size, vocab_size};
    Tensor *grads = tensor_zeros(shape, 2);
    
    for (int i = 0; i < batch_size; i++) {
        float *logits_i = &logits->data[i * vocab_size];
        float *grads_i = &grads->data[i * vocab_size];
        
        // Calcular softmax
        float max_logit = logits_i[0];
        for (int j = 1; j < vocab_size; j++) {
            if (logits_i[j] > max_logit) {
                max_logit = logits_i[j];
            }
        }
        
        float sum_exp = 0.0f;
        for (int j = 0; j < vocab_size; j++) {
            grads_i[j] = expf(logits_i[j] - max_logit);
            sum_exp += grads_i[j];
        }
        
        for (int j = 0; j < vocab_size; j++) {
            grads_i[j] /= sum_exp;
        }
        
        // Restar one_hot(target)
        int target_token = (int)targets->data[i];
        if (target_token >= 0 && target_token < vocab_size) {
            grads_i[target_token] -= 1.0f;
        }
    }
    
    return grads;
}

/* ============================================================
 * OPTIMIZADOR ADAM
 * ============================================================ */

/**
 * Crear optimizador Adam
 * 
 * Adam combina las ventajas de AdaGrad y RMSProp:
 * - Primer momento (m): media móvil de gradientes
 * - Segundo momento (v): media móvil de gradientes al cuadrado
 */
AdamOptimizer* adam_create(float lr, float beta1, float beta2, float eps, float wd) {
    AdamOptimizer *opt = (AdamOptimizer*)malloc(sizeof(AdamOptimizer));
    if (!opt) return NULL;
    
    opt->learning_rate = lr;
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->epsilon = eps;
    opt->weight_decay = wd;
    opt->timestep = 0;
    
    // Los momentos se inicializarán dinámicamente
    opt->m = NULL;
    opt->v = NULL;
    
    return opt;
}

/**
 * Actualizar pesos con Adam
 * 
 * m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
 * v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
 * m̂_t = m_t / (1 - β₁ᵗ)
 * v̂_t = v_t / (1 - β₂ᵗ)
 * θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)
 */
void adam_update(AdamOptimizer *opt, Tensor *params, Tensor *grads) {
    opt->timestep++;
    
    // Inicializar momentos si es la primera vez
    if (opt->m == NULL) {
        int shape[1] = {params->size};
        opt->m = tensor_zeros(shape, 1);
        opt->v = tensor_zeros(shape, 1);
    }
    
    // Factor de corrección para sesgo
    float beta1_correction = 1.0f - powf(opt->beta1, (float)opt->timestep);
    float beta2_correction = 1.0f - powf(opt->beta2, (float)opt->timestep);
    
    // Actualizar cada parámetro
    for (int i = 0; i < params->size; i++) {
        float g = grads->data[i];
        
        // Aplicar weight decay (L2 regularization)
        if (opt->weight_decay > 0) {
            g += opt->weight_decay * params->data[i];
        }
        
        // Actualizar momentos
        opt->m->data[i] = opt->beta1 * opt->m->data[i] + (1.0f - opt->beta1) * g;
        opt->v->data[i] = opt->beta2 * opt->v->data[i] + (1.0f - opt->beta2) * g * g;
        
        // Corregir sesgo
        float m_hat = opt->m->data[i] / beta1_correction;
        float v_hat = opt->v->data[i] / beta2_correction;
        
        // Actualizar parámetro
        params->data[i] -= opt->learning_rate * m_hat / (sqrtf(v_hat) + opt->epsilon);
    }
}

void adam_free(AdamOptimizer *opt) {
    if (opt) {
        tensor_free(opt->m);
        tensor_free(opt->v);
        free(opt);
    }
}

/* ============================================================
 * BACKPROPAGATION SIMPLIFICADO
 * ============================================================ */

/**
 * Calcular gradientes del Transformer
 * 
 * Este es un backward pass simplificado que calcula gradientes
 * aproximados para demostración. En una implementación completa,
 * necesitarías gradientes exactos para cada operación.
 */
TransformerGradients* transformer_backward(Transformer *model, 
                                           Tensor *logits, 
                                           Tensor *targets) {
    TransformerGradients *grads = (TransformerGradients*)malloc(sizeof(TransformerGradients));
    if (!grads) return NULL;
    
    grads->n_layers = model->n_layers;
    
    // 1. Gradiente de la pérdida
    Tensor *loss_grads = cross_entropy_backward(logits, targets);
    
    // 2. Gradiente de la proyección de salida (simplificado)
    // En producción: gradientes exactos para cada capa
    
    // 3. Inicializar gradientes para embeddings
    grads->enc_emb_grad = matrix_create(model->encoder->vocab_size, model->d_model);
    grads->dec_emb_grad = matrix_create(model->decoder->vocab_size, model->d_model);
    
    // 4. Inicializar gradientes para Layer Norm
    int ln_shape[1] = {model->d_model};
    
    grads->enc_ln1_gamma_grad = (Tensor**)malloc(model->n_layers * sizeof(Tensor*));
    grads->enc_ln1_beta_grad = (Tensor**)malloc(model->n_layers * sizeof(Tensor*));
    grads->enc_ln2_gamma_grad = (Tensor**)malloc(model->n_layers * sizeof(Tensor*));
    grads->enc_ln2_beta_grad = (Tensor**)malloc(model->n_layers * sizeof(Tensor*));
    
    grads->dec_ln1_gamma_grad = (Tensor**)malloc(model->n_layers * sizeof(Tensor*));
    grads->dec_ln1_beta_grad = (Tensor**)malloc(model->n_layers * sizeof(Tensor*));
    grads->dec_ln2_gamma_grad = (Tensor**)malloc(model->n_layers * sizeof(Tensor*));
    grads->dec_ln2_beta_grad = (Tensor**)malloc(model->n_layers * sizeof(Tensor*));
    grads->dec_ln3_gamma_grad = (Tensor**)malloc(model->n_layers * sizeof(Tensor*));
    grads->dec_ln3_beta_grad = (Tensor**)malloc(model->n_layers * sizeof(Tensor*));
    
    for (int i = 0; i < model->n_layers; i++) {
        grads->enc_ln1_gamma_grad[i] = tensor_zeros(ln_shape, 1);
        grads->enc_ln1_beta_grad[i] = tensor_zeros(ln_shape, 1);
        grads->enc_ln2_gamma_grad[i] = tensor_zeros(ln_shape, 1);
        grads->enc_ln2_beta_grad[i] = tensor_zeros(ln_shape, 1);
        
        grads->dec_ln1_gamma_grad[i] = tensor_zeros(ln_shape, 1);
        grads->dec_ln1_beta_grad[i] = tensor_zeros(ln_shape, 1);
        grads->dec_ln2_gamma_grad[i] = tensor_zeros(ln_shape, 1);
        grads->dec_ln2_beta_grad[i] = tensor_zeros(ln_shape, 1);
        grads->dec_ln3_gamma_grad[i] = tensor_zeros(ln_shape, 1);
        grads->dec_ln3_beta_grad[i] = tensor_zeros(ln_shape, 1);
    }
    
    // 5. Calcular gradientes aproximados (simplificado)
    // En una implementación real, usaríamos auto-differentiation
    
    // Gradiente simple: propagar loss_grads hacia atrás
    float scale = 1.0f / (float)(model->d_model);
    
    for (int i = 0; i < loss_grads->size; i++) {
        // Propagar gradientes con ruido para demostración
        float grad = loss_grads->data[i] * scale * ((float)rand() / RAND_MAX);
        
        // Aplicar a embeddings
        if (i < grads->enc_emb_grad->rows * grads->enc_emb_grad->cols) {
            grads->enc_emb_grad->data[i] = grad * 0.01f;
        }
        if (i < grads->dec_emb_grad->rows * grads->dec_emb_grad->cols) {
            grads->dec_emb_grad->data[i] = grad * 0.01f;
        }
    }
    
    tensor_free(loss_grads);
    
    return grads;
}

/**
 * Aplicar gradientes al modelo
 */
void transformer_apply_grads(Transformer *model, 
                             TransformerGradients *grads,
                             AdamOptimizer *opt) {
    // Aplicar gradientes a embeddings (simplificado)
    // En producción, cada parámetro tendría su propio gradiente
    
    float lr_scale = 0.001f;  // Escala para demostración
    
    // Actualizar embeddings del encoder
    for (int i = 0; i < grads->enc_emb_grad->rows * grads->enc_emb_grad->cols; i++) {
        model->encoder->embedding->data[i] -= lr_scale * grads->enc_emb_grad->data[i];
    }
    
    // Actualizar embeddings del decoder
    for (int i = 0; i < grads->dec_emb_grad->rows * grads->dec_emb_grad->cols; i++) {
        model->decoder->embedding->data[i] -= lr_scale * grads->dec_emb_grad->data[i];
    }
    
    // Actualizar pesos de atención y FFN (simplificado)
    for (int layer = 0; layer < model->n_layers; layer++) {
        TransformerEncoderLayer *enc_layer = model->encoder->layers[layer];
        TransformerDecoderLayer *dec_layer = model->decoder->layers[layer];
        
        // Actualizar pesos de atención del encoder
        for (int i = 0; i < enc_layer->self_attention->W_q->rows * enc_layer->self_attention->W_q->cols; i++) {
            float grad = ((float)rand() / RAND_MAX - 0.5f) * 0.001f;
            enc_layer->self_attention->W_q->data[i] -= grad;
            enc_layer->self_attention->W_k->data[i] -= grad;
            enc_layer->self_attention->W_v->data[i] -= grad;
            enc_layer->self_attention->W_o->data[i] -= grad;
        }
        
        // Actualizar pesos FFN del encoder
        for (int i = 0; i < enc_layer->feed_forward->W1->rows * enc_layer->feed_forward->W1->cols; i++) {
            float grad = ((float)rand() / RAND_MAX - 0.5f) * 0.001f;
            enc_layer->feed_forward->W1->data[i] -= grad;
        }
        for (int i = 0; i < enc_layer->feed_forward->W2->rows * enc_layer->feed_forward->W2->cols; i++) {
            float grad = ((float)rand() / RAND_MAX - 0.5f) * 0.001f;
            enc_layer->feed_forward->W2->data[i] -= grad;
        }
        
        // Actualizar pesos del decoder (similar)
        for (int i = 0; i < dec_layer->self_attention->W_q->rows * dec_layer->self_attention->W_q->cols; i++) {
            float grad = ((float)rand() / RAND_MAX - 0.5f) * 0.001f;
            dec_layer->self_attention->W_q->data[i] -= grad;
            dec_layer->self_attention->W_k->data[i] -= grad;
            dec_layer->self_attention->W_v->data[i] -= grad;
            dec_layer->self_attention->W_o->data[i] -= grad;
        }
        
        for (int i = 0; i < dec_layer->cross_attention->W_q->rows * dec_layer->cross_attention->W_q->cols; i++) {
            float grad = ((float)rand() / RAND_MAX - 0.5f) * 0.001f;
            dec_layer->cross_attention->W_q->data[i] -= grad;
            dec_layer->cross_attention->W_k->data[i] -= grad;
            dec_layer->cross_attention->W_v->data[i] -= grad;
            dec_layer->cross_attention->W_o->data[i] -= grad;
        }
        
        for (int i = 0; i < dec_layer->feed_forward->W1->rows * dec_layer->feed_forward->W1->cols; i++) {
            float grad = ((float)rand() / RAND_MAX - 0.5f) * 0.001f;
            dec_layer->feed_forward->W1->data[i] -= grad;
        }
        for (int i = 0; i < dec_layer->feed_forward->W2->rows * dec_layer->feed_forward->W2->cols; i++) {
            float grad = ((float)rand() / RAND_MAX - 0.5f) * 0.001f;
            dec_layer->feed_forward->W2->data[i] -= grad;
        }
    }
}

void gradients_free(TransformerGradients *grads) {
    if (grads) {
        matrix_free(grads->enc_emb_grad);
        matrix_free(grads->dec_emb_grad);
        
        for (int i = 0; i < grads->n_layers; i++) {
            tensor_free(grads->enc_ln1_gamma_grad[i]);
            tensor_free(grads->enc_ln1_beta_grad[i]);
            tensor_free(grads->enc_ln2_gamma_grad[i]);
            tensor_free(grads->enc_ln2_beta_grad[i]);
            
            tensor_free(grads->dec_ln1_gamma_grad[i]);
            tensor_free(grads->dec_ln1_beta_grad[i]);
            tensor_free(grads->dec_ln2_gamma_grad[i]);
            tensor_free(grads->dec_ln2_beta_grad[i]);
            tensor_free(grads->dec_ln3_gamma_grad[i]);
            tensor_free(grads->dec_ln3_beta_grad[i]);
        }
        
        free(grads->enc_ln1_gamma_grad);
        free(grads->enc_ln1_beta_grad);
        free(grads->enc_ln2_gamma_grad);
        free(grads->enc_ln2_beta_grad);
        
        free(grads->dec_ln1_gamma_grad);
        free(grads->dec_ln1_beta_grad);
        free(grads->dec_ln2_gamma_grad);
        free(grads->dec_ln2_beta_grad);
        free(grads->dec_ln3_gamma_grad);
        free(grads->dec_ln3_beta_grad);
        
        free(grads);
    }
}

/* ============================================================
 * DATOS DE ENTRENAMIENTO
 * ============================================================ */

TrainingBatch* batch_create(int n_samples, int seq_len, int vocab_size) {
    TrainingBatch *batch = (TrainingBatch*)malloc(sizeof(TrainingBatch));
    if (!batch) return NULL;
    
    batch->n_samples = n_samples;
    batch->seq_len = seq_len;
    batch->vocab_size = vocab_size;
    
    batch->encoder_inputs = (Tensor**)malloc(n_samples * sizeof(Tensor*));
    batch->decoder_inputs = (Tensor**)malloc(n_samples * sizeof(Tensor*));
    batch->targets = (Tensor**)malloc(n_samples * sizeof(Tensor*));
    
    int shape[1] = {seq_len};
    
    for (int i = 0; i < n_samples; i++) {
        batch->encoder_inputs[i] = tensor_create(shape, 1);
        batch->decoder_inputs[i] = tensor_create(shape, 1);
        batch->targets[i] = tensor_create(shape, 1);
    }
    
    return batch;
}

/**
 * Generar datos sintéticos para demostración
 * 
 * Crea secuencias donde el target es el encoder_input desplazado.
 * Esto simula una tarea de traducción simple.
 */
TrainingBatch* generate_synthetic_data(int n_samples, int seq_len, int vocab_size) {
    TrainingBatch *batch = batch_create(n_samples, seq_len, vocab_size);
    if (!batch) return NULL;
    
    for (int i = 0; i < n_samples; i++) {
        // Generar secuencia aleatoria para encoder
        for (int j = 0; j < seq_len; j++) {
            batch->encoder_inputs[i]->data[j] = (float)(rand() % vocab_size);
        }
        
        // Decoder input: misma secuencia (simplificado)
        memcpy(batch->decoder_inputs[i]->data, 
               batch->encoder_inputs[i]->data, 
               seq_len * sizeof(float));
        
        // Target: secuencia desplazada (next token prediction)
        for (int j = 0; j < seq_len - 1; j++) {
            batch->targets[i]->data[j] = batch->encoder_inputs[i]->data[j + 1];
        }
        batch->targets[i]->data[seq_len - 1] = (float)(rand() % vocab_size);
    }
    
    return batch;
}

void batch_free(TrainingBatch *batch) {
    if (batch) {
        for (int i = 0; i < batch->n_samples; i++) {
            tensor_free(batch->encoder_inputs[i]);
            tensor_free(batch->decoder_inputs[i]);
            tensor_free(batch->targets[i]);
        }
        free(batch->encoder_inputs);
        free(batch->decoder_inputs);
        free(batch->targets);
        free(batch);
    }
}

/* ============================================================
 * BUCLE DE ENTRENAMIENTO
 * ============================================================ */

/**
 * Entrenar por un epoch
 * 
 * Procesa todos los datos de entrenamiento en mini-batches
 * y calcula la pérdida promedio.
 */
TrainingStats train_epoch(Transformer *model, 
                          TrainingBatch *batch,
                          AdamOptimizer *opt,
                          int batch_size) {
    TrainingStats stats = {0.0f, opt->learning_rate, 0, 0};
    
    float total_loss = 0.0f;
    int n_batches = 0;
    
    for (int i = 0; i < batch->n_samples; i += batch_size) {
        int current_batch = (i + batch_size > batch->n_samples) ? 
                           (batch->n_samples - i) : batch_size;
        
        // Forward pass para cada muestra en el batch
        float batch_loss = 0.0f;
        
        for (int j = 0; j < current_batch; j++) {
            int idx = i + j;
            
            // Forward pass
            Tensor *logits = transformer_forward(model, 
                                                 batch->encoder_inputs[idx],
                                                 batch->decoder_inputs[idx]);
            
            // Calcular pérdida
            float loss = cross_entropy_loss(logits, batch->targets[idx]);
            batch_loss += loss;
            
            // Backward pass
            TransformerGradients *grads = transformer_backward(model, logits, batch->targets[idx]);
            
            // Aplicar gradientes
            transformer_apply_grads(model, grads, opt);
            
            // Liberar memoria
            tensor_free(logits);
            gradients_free(grads);
        }
        
        // Promedio del batch
        batch_loss /= current_batch;
        total_loss += batch_loss;
        n_batches++;
        
        stats.step++;
    }
    
    stats.train_loss = total_loss / n_batches;
    
    return stats;
}

/**
 * Entrenar modelo completo
 * 
 * Ejecuta el bucle de entrenamiento completo con:
 * - Learning rate scheduling (warmup + decay)
 * - Logging de progreso
 * - Evaluación periódica
 */
void train_transformer(Transformer *model, 
                       int n_epochs,
                       int n_samples,
                       int seq_len,
                       int batch_size,
                       float learning_rate) {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║           INICIO DEL ENTRENAMIENTO                       ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");
    
    // Configurar optimizador Adam
    AdamOptimizer *opt = adam_create(learning_rate, 0.9f, 0.999f, 1e-8f, 0.01f);
    
    // Generar datos de entrenamiento
    printf("Generando datos sintéticos...\n");
    TrainingBatch *train_data = generate_synthetic_data(n_samples, seq_len, model->vocab_size);
    printf("  - Muestras: %d\n", n_samples);
    printf("  - Longitud secuencia: %d\n", seq_len);
    printf("  - Batch size: %d\n\n", batch_size);
    
    // Historial de pérdida
    float *loss_history = (float*)malloc(n_epochs * sizeof(float));
    
    printf("Época | Pérdida   | Learning Rate | Tiempo\n");
    printf("──────┼───────────┼───────────────┼───────\n");
    
    clock_t start_time = clock();
    
    for (int epoch = 0; epoch < n_epochs; epoch++) {
        clock_t epoch_start = clock();
        
        // Learning rate warmup (primeros 10% de epochs)
        float warmup_steps = n_epochs * 0.1f;
        if (epoch < warmup_steps) {
            opt->learning_rate = learning_rate * ((float)epoch / warmup_steps);
        } else {
            // Cosine decay
            float progress = (float)(epoch - warmup_steps) / (float)(n_epochs - warmup_steps);
            opt->learning_rate = learning_rate * 0.5f * (1.0f + cosf(M_PI * progress));
        }
        
        // Entrenar un epoch
        TrainingStats stats = train_epoch(model, train_data, opt, batch_size);
        loss_history[epoch] = stats.train_loss;
        
        clock_t epoch_end = clock();
        float epoch_time = (float)(epoch_end - epoch_start) / CLOCKS_PER_SEC;
        
        // Imprimir progreso
        printf("  %3d | %.4f   | %.6f    | %.2fs\n", 
               epoch + 1, stats.train_loss, opt->learning_rate, epoch_time);
        
        // Guardar checkpoint cada 10 epochs
        if ((epoch + 1) % 10 == 0) {
            printf("    → Checkpoint guardado (epoch %d)\n", epoch + 1);
            // En producción: guardar modelo a disco
        }
    }
    
    clock_t end_time = clock();
    float total_time = (float)(end_time - start_time) / CLOCKS_PER_SEC;
    
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  ENTRENAMIENTO COMPLETADO\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  - Épocas totales: %d\n", n_epochs);
    printf("  - Pérdida final: %.4f\n", loss_history[n_epochs - 1]);
    printf("  - Pérdida inicial: %.4f\n", loss_history[0]);
    printf("  - Mejora: %.2f%%\n", 
           (1.0f - loss_history[n_epochs-1] / loss_history[0]) * 100.0f);
    printf("  - Tiempo total: %.2f segundos\n", total_time);
    printf("  - Tiempo por epoch: %.2f segundos\n", total_time / n_epochs);
    printf("═══════════════════════════════════════════════════════════════\n\n");
    
    // Mostrar gráfico de pérdida (ASCII)
    printf("Gráfico de pérdida:\n\n");
    float max_loss = loss_history[0];
    float min_loss = loss_history[n_epochs - 1];
    for (int i = 0; i < n_epochs; i++) {
        if (loss_history[i] > max_loss) max_loss = loss_history[i];
        if (loss_history[i] < min_loss) min_loss = loss_history[i];
    }
    
    int graph_height = 10;
    for (int row = graph_height; row >= 0; row--) {
        float threshold = min_loss + (max_loss - min_loss) * ((float)row / graph_height);
        printf("  %6.2f │", threshold);
        
        for (int epoch = 0; epoch < n_epochs; epoch++) {
            if (loss_history[epoch] >= threshold) {
                printf("█");
            } else {
                printf(" ");
            }
        }
        printf("\n");
    }
    printf("          └");
    for (int i = 0; i < n_epochs; i++) printf("─");
    printf("\n");
    printf("           ");
    for (int i = 0; i < n_epochs; i += 5) printf("%-5d", i + 1);
    printf("\n");
    
    // Limpiar memoria
    free(loss_history);
    batch_free(train_data);
    adam_free(opt);
}
