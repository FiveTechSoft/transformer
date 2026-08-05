/**
 * TRANSFORMER.H - Arquitectura Transformer Completa en C
 * 
 * Implementación de la arquitectura Transformer original
 * (Attention Is All You Need) con las siguientes componentes:
 * 
 * - Multi-Head Self-Attention
 * - Position-wise Feed-Forward Networks
 * - Positional Encoding
 * - Encoder/Decoder Stack
 * 
 * Esta implementación utiliza operaciones tensoriales para
 * demostrar cómo la lógica y el cálculo se unen a través
 * de representaciones tensoriales.
 */

#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "tensor.h"

/* ============================================================
 * CONSTANTES DEL TRANSFORMER
 * ============================================================ */

#define MAX_SEQ_LEN 512
#define DEFAULT_D_MODEL 512
#define DEFAULT_N_HEADS 8
#define DEFAULT_D_FF 2048
#define DEFAULT_N_LAYERS 6
#define EPSILON 1e-6f

/* ============================================================
 * ESTRUCTURAS DE COMPONENTES
 * ============================================================ */

/**
 * Multi-Head Attention
 * Implementa la atención multi-cabeza que permite al modelo
 * prestar atención a diferentes partes de la secuencia simultáneamente.
 */
typedef struct {
    int d_model;        // Dimensión del modelo
    int n_heads;        // Número de cabezas de atención
    int d_k;            // Dimensión por cabeza (d_model / n_heads)
    
    // Pesos de proyección para Q, K, V
    Matrix *W_q;        // Proyección Query
    Matrix *W_k;        // Proyección Key
    Matrix *W_v;        // Proyección Value
    
    // Pesos de proyección de salida
    Matrix *W_o;        // Proyección de salida
} MultiHeadAttention;

/**
 * Position-wise Feed-Forward Network
 * Red feed-forward aplicada posición por posición
 */
typedef struct {
    int d_model;        // Dimensión de entrada
    int d_ff;           // Dimensión oculta
    
    Matrix *W1;         // Primera capa lineal
    Tensor *b1;         // Bias primera capa
    
    Matrix *W2;         // Segunda capa lineal
    Tensor *b2;         // Bias segunda capa
} FeedForward;

/**
 * Positional Encoding
 * Codificación posicional sinusoidal para inyectar
 * información de orden en la secuencia
 */
typedef struct {
    int d_model;        // Dimensión del modelo
    int max_len;        // Longitud máxima de secuencia
    Tensor *pe;         // Matriz de codificación posicional
} PositionalEncoding;

/**
 * Transformer Layer (Encoder)
 * Una capa completa del encoder
 */
typedef struct {
    MultiHeadAttention *self_attention;
    FeedForward *feed_forward;
    
    // Parámetros de Layer Normalization
    Tensor *ln1_gamma;
    Tensor *ln1_beta;
    Tensor *ln2_gamma;
    Tensor *ln2_beta;
} TransformerEncoderLayer;

/**
 * Transformer Layer (Decoder)
 * Una capa completa del decoder
 */
typedef struct {
    MultiHeadAttention *self_attention;
    MultiHeadAttention *cross_attention;
    FeedForward *feed_forward;
    
    // Parámetros de Layer Normalization
    Tensor *ln1_gamma;
    Tensor *ln1_beta;
    Tensor *ln2_gamma;
    Tensor *ln2_beta;
    Tensor *ln3_gamma;
    Tensor *ln3_beta;
} TransformerDecoderLayer;

/**
 * Transformer Encoder
 * Stack de capas encoder
 */
typedef struct {
    int n_layers;
    PositionalEncoding *pos_encoding;
    TransformerEncoderLayer **layers;
    
    // Parámetros de embedding
    Matrix *embedding;
    int vocab_size;
    int d_model;
} TransformerEncoder;

/**
 * Transformer Decoder
 * Stack de capas decoder
 */
typedef struct {
    int n_layers;
    PositionalEncoding *pos_encoding;
    TransformerDecoderLayer **layers;
    
    // Parámetros de embedding
    Matrix *embedding;
    Matrix *output_projection;
    int vocab_size;
    int d_model;
} TransformerDecoder;

/**
 * Transformer Completo
 * Arquitectura Encoder-Decoder completa
 */
typedef struct {
    TransformerEncoder *encoder;
    TransformerDecoder *decoder;
    
    int vocab_size;
    int d_model;
    int n_heads;
    int d_ff;
    int n_layers;
} Transformer;

/* ============================================================
 * FUNCIONES DE CREACIÓN Y DESTRUCCIÓN
 * ============================================================ */

// Crear Multi-Head Attention
MultiHeadAttention* mha_create(int d_model, int n_heads);

// Crear Feed-Forward Network
FeedForward* ff_create(int d_model, int d_ff);

// Crear Positional Encoding
PositionalEncoding* pe_create(int d_model, int max_len);

// Crear Transformer Encoder Layer
TransformerEncoderLayer* encoder_layer_create(int d_model, int n_heads, int d_ff);

// Crear Transformer Decoder Layer
TransformerDecoderLayer* decoder_layer_create(int d_model, int n_heads, int d_ff);

// Crear Transformer Encoder
TransformerEncoder* encoder_create(int vocab_size, int d_model, int n_heads, int d_ff, int n_layers);

// Crear Transformer Decoder
TransformerDecoder* decoder_create(int vocab_size, int d_model, int n_heads, int d_ff, int n_layers);

// Crear Transformer completo
Transformer* transformer_create(int vocab_size, int d_model, int n_heads, int d_ff, int n_layers);

// Liberar memoria
void mha_free(MultiHeadAttention *mha);
void ff_free(FeedForward *ff);
void pe_free(PositionalEncoding *pe);
void encoder_layer_free(TransformerEncoderLayer *layer);
void decoder_layer_free(TransformerDecoderLayer *layer);
void encoder_free(TransformerEncoder *encoder);
void decoder_free(TransformerDecoder *decoder);
void transformer_free(Transformer *transformer);

/* ============================================================
 * FUNCIONES FORWARD
 * ============================================================ */

// Multi-Head Attention forward
Tensor* mha_forward(MultiHeadAttention *mha, Tensor *q, Tensor *k, Tensor *v);

// Feed-Forward forward
Tensor* ff_forward(FeedForward *ff, Tensor *x);

// Encoder Layer forward
Tensor* encoder_layer_forward(TransformerEncoderLayer *layer, Tensor *x);

// Decoder Layer forward
Tensor* decoder_layer_forward(TransformerDecoderLayer *layer, Tensor *x, 
                              Tensor *encoder_output);

// Encoder forward
Tensor* encoder_forward(TransformerEncoder *encoder, Tensor *x);

// Decoder forward
Tensor* decoder_forward(TransformerDecoder *decoder, Tensor *x, 
                        Tensor *encoder_output);

// Transformer forward completo
Tensor* transformer_forward(Transformer *transformer, Tensor *encoder_input,
                            Tensor *decoder_input);

/* ============================================================
 * FUNCIONES DE UTILIDAD
 * ============================================================ */

// Aplicar embedding
Matrix* apply_embedding(Matrix *embedding, Tensor *input_ids);

// Generar máscara dePadding
Tensor* create_padding_mask(Tensor *input, int pad_idx);

// Generar máscara causal (look-ahead)
Tensor* create_causal_mask(int seq_len);

// Imprimir estructura del Transformer
void transformer_print_info(Transformer *transformer);

#endif /* TRANSFORMER_H */
