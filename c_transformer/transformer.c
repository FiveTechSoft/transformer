/**
 * TRANSFORMER.C - Implementación del Transformer
 * 
 * Implementación completa de la arquitectura Transformer
 * basada en "Attention Is All You Need" (Vaswani et al., 2017).
 * 
 * Esta implementación demuestra cómo las operaciones tensoriales
 * permiten construir modelos de lenguaje sofisticados que capturan
 * relaciones de largo alcance en secuencias.
 */

#include "transformer.h"

/* ============================================================
 * FUNCIONES DE CREACIÓN
 * ============================================================ */

MultiHeadAttention* mha_create(int d_model, int n_heads) {
    MultiHeadAttention *mha = (MultiHeadAttention*)malloc(sizeof(MultiHeadAttention));
    if (!mha) return NULL;
    
    mha->d_model = d_model;
    mha->n_heads = n_heads;
    mha->d_k = d_model / n_heads;
    
    // Inicializar pesos de proyección
    int shape_qkv[2] = {d_model, d_model};
    mha->W_q = matrix_create(d_model, d_model);
    mha->W_k = matrix_create(d_model, d_model);
    mha->W_v = matrix_create(d_model, d_model);
    mha->W_o = matrix_create(d_model, d_model);
    
    // Inicialización Xavier para mejor convergencia
    float limit = sqrt(6.0f / (float)d_model);
    for (int i = 0; i < d_model * d_model; i++) {
        mha->W_q->data[i] = ((float)rand() / RAND_MAX) * 2.0f * limit - limit;
        mha->W_k->data[i] = ((float)rand() / RAND_MAX) * 2.0f * limit - limit;
        mha->W_v->data[i] = ((float)rand() / RAND_MAX) * 2.0f * limit - limit;
        mha->W_o->data[i] = ((float)rand() / RAND_MAX) * 2.0f * limit - limit;
    }
    
    return mha;
}

FeedForward* ff_create(int d_model, int d_ff) {
    FeedForward *ff = (FeedForward*)malloc(sizeof(FeedForward));
    if (!ff) return NULL;
    
    ff->d_model = d_model;
    ff->d_ff = d_ff;
    
    // Primera capa lineal: d_model -> d_ff
    ff->W1 = matrix_create(d_model, d_ff);
    int shape_b1[1] = {d_ff};
    ff->b1 = tensor_create(shape_b1, 1);
    
    // Segunda capa lineal: d_ff -> d_model
    ff->W2 = matrix_create(d_ff, d_model);
    int shape_b2[1] = {d_model};
    ff->b2 = tensor_create(shape_b2, 1);
    
    // Inicialización Xavier
    float limit1 = sqrt(6.0f / (float)d_model);
    float limit2 = sqrt(6.0f / (float)d_ff);
    
    for (int i = 0; i < d_model * d_ff; i++) {
        ff->W1->data[i] = ((float)rand() / RAND_MAX) * 2.0f * limit1 - limit1;
    }
    for (int i = 0; i < d_ff * d_model; i++) {
        ff->W2->data[i] = ((float)rand() / RAND_MAX) * 2.0f * limit2 - limit2;
    }
    
    return ff;
}

PositionalEncoding* pe_create(int d_model, int max_len) {
    PositionalEncoding *pe = (PositionalEncoding*)malloc(sizeof(PositionalEncoding));
    if (!pe) return NULL;
    
    pe->d_model = d_model;
    pe->max_len = max_len;
    
    int shape[2] = {max_len, d_model};
    pe->pe = tensor_create(shape, 2);
    
    // Generar codificación posicional sinusoidal
    for (int pos = 0; pos < max_len; pos++) {
        for (int i = 0; i < d_model; i++) {
            float angle = (float)pos / powf(10000.0f, (float)(2 * (i / 2)) / (float)d_model);
            
            if (i % 2 == 0) {
                pe->pe->data[pos * d_model + i] = sinf(angle);
            } else {
                pe->pe->data[pos * d_model + i] = cosf(angle);
            }
        }
    }
    
    return pe;
}

TransformerEncoderLayer* encoder_layer_create(int d_model, int n_heads, int d_ff) {
    TransformerEncoderLayer *layer = (TransformerEncoderLayer*)malloc(sizeof(TransformerEncoderLayer));
    if (!layer) return NULL;
    
    layer->self_attention = mha_create(d_model, n_heads);
    layer->feed_forward = ff_create(d_model, d_ff);
    
    // Parámetros de Layer Normalization
    int shape_ln[1] = {d_model};
    layer->ln1_gamma = tensor_ones(shape_ln, 1);
    layer->ln1_beta = tensor_zeros(shape_ln, 1);
    layer->ln2_gamma = tensor_ones(shape_ln, 1);
    layer->ln2_beta = tensor_zeros(shape_ln, 1);
    
    return layer;
}

TransformerDecoderLayer* decoder_layer_create(int d_model, int n_heads, int d_ff) {
    TransformerDecoderLayer *layer = (TransformerDecoderLayer*)malloc(sizeof(TransformerDecoderLayer));
    if (!layer) return NULL;
    
    layer->self_attention = mha_create(d_model, n_heads);
    layer->cross_attention = mha_create(d_model, n_heads);
    layer->feed_forward = ff_create(d_model, d_ff);
    
    // Parámetros de Layer Normalization
    int shape_ln[1] = {d_model};
    layer->ln1_gamma = tensor_ones(shape_ln, 1);
    layer->ln1_beta = tensor_zeros(shape_ln, 1);
    layer->ln2_gamma = tensor_ones(shape_ln, 1);
    layer->ln2_beta = tensor_zeros(shape_ln, 1);
    layer->ln3_gamma = tensor_ones(shape_ln, 1);
    layer->ln3_beta = tensor_zeros(shape_ln, 1);
    
    return layer;
}

TransformerEncoder* encoder_create(int vocab_size, int d_model, int n_heads, int d_ff, int n_layers) {
    TransformerEncoder *encoder = (TransformerEncoder*)malloc(sizeof(TransformerEncoder));
    if (!encoder) return NULL;
    
    encoder->vocab_size = vocab_size;
    encoder->d_model = d_model;
    encoder->n_layers = n_layers;
    
    // Crear embedding
    encoder->embedding = matrix_create(vocab_size, d_model);
    float limit = sqrt(6.0f / (float)(vocab_size + d_model));
    for (int i = 0; i < vocab_size * d_model; i++) {
        encoder->embedding->data[i] = ((float)rand() / RAND_MAX) * 2.0f * limit - limit;
    }
    
    // Crear positional encoding
    encoder->pos_encoding = pe_create(d_model, MAX_SEQ_LEN);
    
    // Crear capas
    encoder->layers = (TransformerEncoderLayer**)malloc(n_layers * sizeof(TransformerEncoderLayer*));
    for (int i = 0; i < n_layers; i++) {
        encoder->layers[i] = encoder_layer_create(d_model, n_heads, d_ff);
    }
    
    return encoder;
}

TransformerDecoder* decoder_create(int vocab_size, int d_model, int n_heads, int d_ff, int n_layers) {
    TransformerDecoder *decoder = (TransformerDecoder*)malloc(sizeof(TransformerDecoder));
    if (!decoder) return NULL;
    
    decoder->vocab_size = vocab_size;
    decoder->d_model = d_model;
    decoder->n_layers = n_layers;
    
    // Crear embedding
    decoder->embedding = matrix_create(vocab_size, d_model);
    float limit = sqrt(6.0f / (float)(vocab_size + d_model));
    for (int i = 0; i < vocab_size * d_model; i++) {
        decoder->embedding->data[i] = ((float)rand() / RAND_MAX) * 2.0f * limit - limit;
    }
    
    // Crear proyección de salida
    decoder->output_projection = matrix_create(d_model, vocab_size);
    limit = sqrt(6.0f / (float)(d_model + vocab_size));
    for (int i = 0; i < d_model * vocab_size; i++) {
        decoder->output_projection->data[i] = ((float)rand() / RAND_MAX) * 2.0f * limit - limit;
    }
    
    // Crear positional encoding
    decoder->pos_encoding = pe_create(d_model, MAX_SEQ_LEN);
    
    // Crear capas
    decoder->layers = (TransformerDecoderLayer**)malloc(n_layers * sizeof(TransformerDecoderLayer*));
    for (int i = 0; i < n_layers; i++) {
        decoder->layers[i] = decoder_layer_create(d_model, n_heads, d_ff);
    }
    
    return decoder;
}

Transformer* transformer_create(int vocab_size, int d_model, int n_heads, int d_ff, int n_layers) {
    Transformer *transformer = (Transformer*)malloc(sizeof(Transformer));
    if (!transformer) return NULL;
    
    transformer->vocab_size = vocab_size;
    transformer->d_model = d_model;
    transformer->n_heads = n_heads;
    transformer->d_ff = d_ff;
    transformer->n_layers = n_layers;
    
    // Inicializar semilla aleatoria
    tensor_seed(time(NULL));
    
    // Crear encoder y decoder
    transformer->encoder = encoder_create(vocab_size, d_model, n_heads, d_ff, n_layers);
    transformer->decoder = decoder_create(vocab_size, d_model, n_heads, d_ff, n_layers);
    
    return transformer;
}

/* ============================================================
 * FUNCIONES DE LIBERACIÓN DE MEMORIA
 * ============================================================ */

void mha_free(MultiHeadAttention *mha) {
    if (mha) {
        matrix_free(mha->W_q);
        matrix_free(mha->W_k);
        matrix_free(mha->W_v);
        matrix_free(mha->W_o);
        free(mha);
    }
}

void ff_free(FeedForward *ff) {
    if (ff) {
        matrix_free(ff->W1);
        tensor_free(ff->b1);
        matrix_free(ff->W2);
        tensor_free(ff->b2);
        free(ff);
    }
}

void pe_free(PositionalEncoding *pe) {
    if (pe) {
        tensor_free(pe->pe);
        free(pe);
    }
}

void encoder_layer_free(TransformerEncoderLayer *layer) {
    if (layer) {
        mha_free(layer->self_attention);
        ff_free(layer->feed_forward);
        tensor_free(layer->ln1_gamma);
        tensor_free(layer->ln1_beta);
        tensor_free(layer->ln2_gamma);
        tensor_free(layer->ln2_beta);
        free(layer);
    }
}

void decoder_layer_free(TransformerDecoderLayer *layer) {
    if (layer) {
        mha_free(layer->self_attention);
        mha_free(layer->cross_attention);
        ff_free(layer->feed_forward);
        tensor_free(layer->ln1_gamma);
        tensor_free(layer->ln1_beta);
        tensor_free(layer->ln2_gamma);
        tensor_free(layer->ln2_beta);
        tensor_free(layer->ln3_gamma);
        tensor_free(layer->ln3_beta);
        free(layer);
    }
}

void encoder_free(TransformerEncoder *encoder) {
    if (encoder) {
        matrix_free(encoder->embedding);
        pe_free(encoder->pos_encoding);
        for (int i = 0; i < encoder->n_layers; i++) {
            encoder_layer_free(encoder->layers[i]);
        }
        free(encoder->layers);
        free(encoder);
    }
}

void decoder_free(TransformerDecoder *decoder) {
    if (decoder) {
        matrix_free(decoder->embedding);
        matrix_free(decoder->output_projection);
        pe_free(decoder->pos_encoding);
        for (int i = 0; i < decoder->n_layers; i++) {
            decoder_layer_free(decoder->layers[i]);
        }
        free(decoder->layers);
        free(decoder);
    }
}

void transformer_free(Transformer *transformer) {
    if (transformer) {
        encoder_free(transformer->encoder);
        decoder_free(transformer->decoder);
        free(transformer);
    }
}

/* ============================================================
 * FUNCIONES FORWARD (PASO ADELANTE)
 * ============================================================ */

/**
 * Multi-Head Attention Forward
 * 
 * Implementa el mecanismo de atención multi-cabeza:
 * 1. Proyectar Q, K, V
 * 2. Dividir en cabezas
 * 3. Calcular atención para cada cabeza
 * 4. Concatenar resultados
 * 5. Proyectar salida
 */
Tensor* mha_forward(MultiHeadAttention *mha, Tensor *q, Tensor *k, Tensor *v) {
    int seq_len = q->shape[0];
    int d_model = mha->d_model;
    int n_heads = mha->n_heads;
    int d_k = mha->d_k;
    
    // Proyectar Q, K, V usando operaciones matriciales
    Matrix *Q = matrix_multiply(matrix_create_from_data(q->data, seq_len, d_model), mha->W_q);
    Matrix *K = matrix_multiply(matrix_create_from_data(k->data, seq_len, d_model), mha->W_k);
    Matrix *V = matrix_multiply(matrix_create_from_data(v->data, seq_len, d_model), mha->W_v);
    
    // Calcular puntuaciones de atención: scores = Q * K^T / sqrt(d_k)
    Matrix *K_T = matrix_transpose(K);
    Matrix *scores = matrix_multiply(Q, K_T);
    
    float scale = 1.0f / sqrtf((float)d_k);
    scores = matrix_scale(scores, scale);
    
    // Aplicar softmax (simplificado: aplicamos a cada fila)
    for (int i = 0; i < scores->rows; i++) {
        // Encontrar max para estabilidad numérica
        float max_val = scores->data[i * scores->cols];
        for (int j = 1; j < scores->cols; j++) {
            if (scores->data[i * scores->cols + j] > max_val) {
                max_val = scores->data[i * scores->cols + j];
            }
        }
        
        // Calcular softmax
        float sum = 0.0f;
        for (int j = 0; j < scores->cols; j++) {
            scores->data[i * scores->cols + j] = expf(scores->data[i * scores->cols + j] - max_val);
            sum += scores->data[i * scores->cols + j];
        }
        for (int j = 0; j < scores->cols; j++) {
            scores->data[i * scores->cols + j] /= sum;
        }
    }
    
    // Calcular salida: attention = scores * V
    Matrix *attention = matrix_multiply(scores, V);
    
    // Proyectar salida final
    Matrix *output = matrix_multiply(attention, mha->W_o);
    
    // Convertir resultado a tensor
    int shape[2] = {seq_len, d_model};
    Tensor *result = tensor_create_from_data(output->data, shape, 2);
    
    // Liberar memoria temporal
    matrix_free(Q);
    matrix_free(K);
    matrix_free(V);
    matrix_free(K_T);
    matrix_free(scores);
    matrix_free(attention);
    matrix_free(output);
    
    return result;
}

/**
 * Feed-Forward Network Forward
 * 
 * Aplica la red feed-forward posición por posición:
 * FFN(x) = max(0, xW1 + b1)W2 + b2
 * Usamos GELU en lugar de ReLU (como en BERT/GPT)
 */
Tensor* ff_forward(FeedForward *ff, Tensor *x) {
    int seq_len = x->shape[0];
    int d_model = ff->d_model;
    int d_ff = ff->d_ff;
    
    // Primera capa: x * W1 + b1
    Matrix *X = matrix_create_from_data(x->data, seq_len, d_model);
    Matrix *h = matrix_multiply(X, ff->W1);
    
    // Agregar bias (simplificado: replicamos para cada posición)
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_ff; j++) {
            h->data[i * d_ff + j] += ff->b1->data[j];
        }
    }
    
    // Aplicar GELU
    for (int i = 0; i < h->rows * h->cols; i++) {
        float val = h->data[i];
        float inner = sqrtf(2.0f / M_PI) * (val + 0.044715f * val * val * val);
        h->data[i] = 0.5f * val * (1.0f + tanhf(inner));
    }
    
    // Segunda capa: h * W2 + b2
    Matrix *output = matrix_multiply(h, ff->W2);
    
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model; j++) {
            output->data[i * d_model + j] += ff->b2->data[j];
        }
    }
    
    // Convertir a tensor
    int shape[2] = {seq_len, d_model};
    Tensor *result = tensor_create_from_data(output->data, shape, 2);
    
    matrix_free(X);
    matrix_free(h);
    matrix_free(output);
    
    return result;
}

/**
 * Encoder Layer Forward
 * 
 * Implementa una capa completa del encoder:
 * 1. Multi-Head Self-Attention
 * 2. Add & Norm
 * 3. Feed-Forward
 * 4. Add & Norm
 */
Tensor* encoder_layer_forward(TransformerEncoderLayer *layer, Tensor *x) {
    // Self-Attention
    Tensor *attn_output = mha_forward(layer->self_attention, x, x, x);
    
    // Add & Norm (primera subcapa)
    Tensor *norm1 = tensor_layer_norm(tensor_add(x, attn_output), 
                                       layer->ln1_gamma, layer->ln1_beta);
    
    // Feed-Forward
    Tensor *ff_output = ff_forward(layer->feed_forward, norm1);
    
    // Add & Norm (segunda subcapa)
    Tensor *norm2 = tensor_layer_norm(tensor_add(norm1, ff_output),
                                       layer->ln2_gamma, layer->ln2_beta);
    
    tensor_free(attn_output);
    tensor_free(norm1);
    tensor_free(ff_output);
    
    return norm2;
}

/**
 * Decoder Layer Forward
 * 
 * Implementa una capa completa del decoder:
 * 1. Masked Multi-Head Self-Attention
 * 2. Add & Norm
 * 3. Multi-Head Cross-Attention
 * 4. Add & Norm
 * 5. Feed-Forward
 * 6. Add & Norm
 */
Tensor* decoder_layer_forward(TransformerDecoderLayer *layer, Tensor *x, 
                              Tensor *encoder_output) {
    // Masked Self-Attention (simplificado: sin máscara aquí)
    Tensor *self_attn_output = mha_forward(layer->self_attention, x, x, x);
    
    // Add & Norm
    Tensor *norm1 = tensor_layer_norm(tensor_add(x, self_attn_output),
                                       layer->ln1_gamma, layer->ln1_beta);
    
    // Cross-Attention (Q del decoder, K y V del encoder)
    Tensor *cross_attn_output = mha_forward(layer->cross_attention, 
                                             norm1, encoder_output, encoder_output);
    
    // Add & Norm
    Tensor *norm2 = tensor_layer_norm(tensor_add(norm1, cross_attn_output),
                                       layer->ln2_gamma, layer->ln2_beta);
    
    // Feed-Forward
    Tensor *ff_output = ff_forward(layer->feed_forward, norm2);
    
    // Add & Norm
    Tensor *norm3 = tensor_layer_norm(tensor_add(norm2, ff_output),
                                       layer->ln3_gamma, layer->ln3_beta);
    
    tensor_free(self_attn_output);
    tensor_free(norm1);
    tensor_free(cross_attn_output);
    tensor_free(norm2);
    tensor_free(ff_output);
    
    return norm3;
}

/**
 * Encoder Forward
 * 
 * Procesa la secuencia de entrada completa a través del encoder
 */
Tensor* encoder_forward(TransformerEncoder *encoder, Tensor *x) {
    int seq_len = x->shape[0];
    int d_model = encoder->d_model;
    
    // Obtener embeddings
    Matrix *embedded = apply_embedding(encoder->embedding, x);
    
    // Agregar positional encoding
    Tensor *embed_tensor = tensor_create_from_data(embedded->data, 
                                                    (int[]){seq_len, d_model}, 2);
    
    // Sumar positional encoding
    Tensor *pos = tensor_add(embed_tensor, 
                              tensor_create_from_data(
                                  &encoder->pos_encoding->pe->data[0],
                                  (int[]){seq_len, d_model}, 2));
    
    // Escalar por sqrt(d_model) (como en el paper original)
    pos = tensor_scale(pos, sqrtf((float)d_model));
    
    // Pasar por cada capa del encoder
    Tensor *current = pos;
    for (int i = 0; i < encoder->n_layers; i++) {
        Tensor *next = encoder_layer_forward(encoder->layers[i], current);
        if (i > 0) tensor_free(current);
        current = next;
    }
    
    matrix_free(embedded);
    tensor_free(embed_tensor);
    
    return current;
}

/**
 * Decoder Forward
 * 
 * Procesa la secuencia objetivo a través del decoder
 */
Tensor* decoder_forward(TransformerDecoder *decoder, Tensor *x, 
                        Tensor *encoder_output) {
    int seq_len = x->shape[0];
    int d_model = decoder->d_model;
    
    // Obtener embeddings
    Matrix *embedded = apply_embedding(decoder->embedding, x);
    
    // Agregar positional encoding
    Tensor *embed_tensor = tensor_create_from_data(embedded->data,
                                                    (int[]){seq_len, d_model}, 2);
    
    Tensor *pos = tensor_add(embed_tensor,
                              tensor_create_from_data(
                                  &decoder->pos_encoding->pe->data[0],
                                  (int[]){seq_len, d_model}, 2));
    
    pos = tensor_scale(pos, sqrtf((float)d_model));
    
    // Pasar por cada capa del decoder
    Tensor *current = pos;
    for (int i = 0; i < decoder->n_layers; i++) {
        Tensor *next = decoder_layer_forward(decoder->layers[i], current, encoder_output);
        if (i > 0) tensor_free(current);
        current = next;
    }
    
    matrix_free(embedded);
    tensor_free(embed_tensor);
    
    return current;
}

/**
 * Transformer Forward
 * 
 * Paso adelante completo del Transformer
 */
Tensor* transformer_forward(Transformer *transformer, Tensor *encoder_input,
                            Tensor *decoder_input) {
    // Encoder
    Tensor *encoder_output = encoder_forward(transformer->encoder, encoder_input);
    
    // Decoder
    Tensor *decoder_output = decoder_forward(transformer->decoder, decoder_input, encoder_output);
    
    // Proyección de salida final (logits sobre vocabulario)
    int seq_len = decoder_output->shape[0];
    int d_model = decoder_output->shape[1];
    
    Matrix *X = matrix_create_from_data(decoder_output->data, seq_len, d_model);
    Matrix *logits = matrix_multiply(X, transformer->decoder->output_projection);
    
    // Convertir a tensor
    int shape[2] = {seq_len, transformer->vocab_size};
    Tensor *result = tensor_create_from_data(logits->data, shape, 2);
    
    tensor_free(encoder_output);
    tensor_free(decoder_output);
    matrix_free(X);
    matrix_free(logits);
    
    return result;
}

/* ============================================================
 * FUNCIONES DE UTILIDAD
 * ============================================================ */

Matrix* apply_embedding(Matrix *embedding, Tensor *input_ids) {
    int seq_len = input_ids->shape[0];
    int d_model = embedding->cols;
    
    Matrix *output = matrix_create(seq_len, d_model);
    
    for (int i = 0; i < seq_len; i++) {
        int idx = (int)input_ids->data[i];
        if (idx >= 0 && idx < embedding->rows) {
            memcpy(&output->data[i * d_model], 
                   &embedding->data[idx * d_model],
                   d_model * sizeof(float));
        }
    }
    
    return output;
}

Tensor* create_padding_mask(Tensor *input, int pad_idx) {
    int seq_len = input->shape[0];
    int shape[2] = {seq_len, seq_len};
    Tensor *mask = tensor_create(shape, 2);
    
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            // Mascarar posiciones de padding
            if ((int)input->data[i] == pad_idx || (int)input->data[j] == pad_idx) {
                mask->data[i * seq_len + j] = -1e9f;
            } else {
                mask->data[i * seq_len + j] = 0.0f;
            }
        }
    }
    
    return mask;
}

Tensor* create_causal_mask(int seq_len) {
    int shape[2] = {seq_len, seq_len};
    Tensor *mask = tensor_create(shape, 2);
    
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            // Mascarar posiciones futuras (j > i)
            if (j > i) {
                mask->data[i * seq_len + j] = -1e9f;
            } else {
                mask->data[i * seq_len + j] = 0.0f;
            }
        }
    }
    
    return mask;
}

void transformer_print_info(Transformer *transformer) {
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║           TRANSFORMER ARCHITECTURE INFO                  ║\n");
    printf("╠════════════════════════════════════════════════════════════╣\n");
    printf("║  Vocab Size:     %-6d                                 ║\n", transformer->vocab_size);
    printf("║  d_model:        %-6d                                 ║\n", transformer->d_model);
    printf("║  Num Heads:      %-6d                                 ║\n", transformer->n_heads);
    printf("║  d_ff:           %-6d                                 ║\n", transformer->d_ff);
    printf("║  Num Layers:     %-6d                                 ║\n", transformer->n_layers);
    printf("║  d_k (per head): %-6d                                 ║\n", transformer->d_model / transformer->n_heads);
    printf("╠════════════════════════════════════════════════════════════╣\n");
    
    // Calcular parámetros totales
    long total_params = 0;
    
    // Embeddings
    total_params += transformer->vocab_size * transformer->d_model * 2; // encoder + decoder
    
    // Por cada capa
    for (int i = 0; i < transformer->n_layers; i++) {
        // Encoder layer: 4 * d_model^2 (attention) + 2 * d_model * d_ff (ff) + 4 * d_model (LN)
        total_params += 4 * transformer->d_model * transformer->d_model;
        total_params += 2 * transformer->d_model * transformer->d_ff;
        total_params += 4 * transformer->d_model;
        
        // Decoder layer: 3 * 4 * d_model^2 (3 attentions) + 2 * d_model * d_ff + 6 * d_model
        total_params += 12 * transformer->d_model * transformer->d_model;
        total_params += 2 * transformer->d_model * transformer->d_ff;
        total_params += 6 * transformer->d_model;
    }
    
    // Output projection
    total_params += transformer->d_model * transformer->vocab_size;
    
    printf("║  Total Parameters: %ld                                ║\n", total_params);
    printf("║  Model Size: ~%.2f MB (float32)                      ║\n", 
           (float)total_params * 4 / (1024 * 1024));
    printf("╚════════════════════════════════════════════════════════════╝\n");
}
