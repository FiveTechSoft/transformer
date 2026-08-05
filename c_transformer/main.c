/**
 * MAIN.C - Ejemplo de uso del Transformer
 * 
 * Este archivo demuestra cómo crear y usar un Transformer
 * para tareas de procesamiento de lenguaje natural.
 * 
 * El Transformer utiliza operaciones tensoriales para
 * procesar secuencias y generar predicciones, demostrando
 * la unión de lógica y cálculo numérico a través de tensores.
 */

#include "transformer.h"

/* ============================================================
 * FUNCIONES DE UTILIDAD PARA DEMOSTRACIÓN
 * ============================================================ */

// Crear secuencia de ejemplo (simulando tokens)
Tensor* create_sample_sequence(int seq_len, int vocab_size) {
    int shape[1] = {seq_len};
    Tensor *seq = tensor_create(shape, 1);
    
    for (int i = 0; i < seq_len; i++) {
        seq->data[i] = (float)(rand() % vocab_size);
    }
    
    return seq;
}

// Imprimir predicciones
void print_predictions(Tensor *logits, int top_k) {
    int seq_len = logits->shape[0];
    int vocab_size = logits->shape[1];
    
    printf("\n=== PREDICCIONES ===\n");
    
    for (int pos = 0; pos < seq_len; pos++) {
        printf("\nPosición %d:\n", pos);
        
        // Encontrar top-k predicciones
        float *scores = &logits->data[pos * vocab_size];
        int *indices = (int*)malloc(vocab_size * sizeof(int));
        
        for (int i = 0; i < vocab_size; i++) {
            indices[i] = i;
        }
        
        // Ordenar por puntuación (simplificado)
        for (int i = 0; i < top_k && i < vocab_size; i++) {
            int max_idx = i;
            for (int j = i + 1; j < vocab_size; j++) {
                if (scores[indices[j]] > scores[indices[max_idx]]) {
                    max_idx = j;
                }
            }
            int temp = indices[i];
            indices[i] = indices[max_idx];
            indices[max_idx] = temp;
            
            // Aplicar softmax para obtener probabilidades
            float max_score = scores[indices[0]];
            float exp_sum = 0.0f;
            for (int j = 0; j < vocab_size; j++) {
                exp_sum += expf(scores[j] - max_score);
            }
            float prob = expf(scores[indices[i]] - max_score) / exp_sum;
            
            printf("  Top %d: Token %d (prob: %.4f)\n", i + 1, indices[i], prob);
        }
        
        free(indices);
    }
}

/* ============================================================
 * EJEMPLO: CLASIFICACIÓN DE SECUENCIAS
 * ============================================================ */

void example_classification() {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║       EJEMPLO: CLASIFICACIÓN DE SECUENCIAS               ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");
    
    // Parámetros reducidos para demostración
    int vocab_size = 1000;
    int d_model = 128;
    int n_heads = 4;
    int d_ff = 256;
    int n_layers = 2;
    int seq_len = 10;
    
    printf("Creando Transformer con parámetros:\n");
    printf("  - Vocabulario: %d tokens\n", vocab_size);
    printf("  - Dimensión del modelo: %d\n", d_model);
    printf("  - Cabezas de atención: %d\n", n_heads);
    printf("  - Dimensión FF: %d\n", d_ff);
    printf("  - Capas: %d\n\n", n_layers);
    
    // Crear transformer
    Transformer *model = transformer_create(vocab_size, d_model, n_heads, d_ff, n_layers);
    transformer_print_info(model);
    
    // Crear secuencia de entrada
    printf("\nCreando secuencia de entrada...\n");
    Tensor *input = create_sample_sequence(seq_len, vocab_size);
    tensor_print(input, "Input Sequence");
    
    // Crear secuencia de salida (para decoder)
    Tensor *target = create_sample_sequence(seq_len, vocab_size);
    tensor_print(target, "Target Sequence");
    
    // Paso adelante
    printf("\nEjecutando forward pass...\n");
    Tensor *output = transformer_forward(model, input, target);
    
    printf("\nDimensiones de salida: [%d, %d]\n", output->shape[0], output->shape[1]);
    
    // Mostrar algunas predicciones
    print_predictions(output, 3);
    
    // Liberar memoria
    tensor_free(input);
    tensor_free(target);
    tensor_free(output);
    transformer_free(model);
    
    printf("\n✓ Ejemplo de clasificación completado.\n");
}

/* ============================================================
 * EJEMPLO: GENERACIÓN DE TEXTO
 * ============================================================ */

void example_text_generation() {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║       EJEMPLO: GENERACIÓN DE TEXTO                       ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");
    
    // Parámetros para generación
    int vocab_size = 5000;
    int d_model = 256;
    int n_heads = 8;
    int d_ff = 512;
    int n_layers = 4;
    
    printf("Creando modelo de generación...\n");
    Transformer *model = transformer_create(vocab_size, d_model, n_heads, d_ff, n_layers);
    transformer_print_info(model);
    
    // Simular generación autoregresiva
    int prompt_len = 5;
    int generated_len = 10;
    
    printf("\nSimulando generación de texto:\n");
    printf("  - Prompt: %d tokens\n", prompt_len);
    printf("  - Generando: %d tokens\n\n", generated_len);
    
    // Crear prompt
    int shape[1] = {prompt_len};
    Tensor *current = tensor_create(shape, 1);
    for (int i = 0; i < prompt_len; i++) {
        current->data[i] = (float)(rand() % 100); // Tokens bajos como "inicio"
    }
    
    printf("Prompt inicial: ");
    for (int i = 0; i < prompt_len; i++) {
        printf("%d ", (int)current->data[i]);
    }
    printf("\n\n");
    
    // Generar tokens (simplificado - en la realidad usaríamos decodificación greedy/beam)
    printf("Generando tokens:\n");
    
    for (int step = 0; step < generated_len; step++) {
        // Crear nueva secuencia con token generado
        int new_len = current->shape[0] + 1;
        int new_shape[1] = {new_len};
        Tensor *new_seq = tensor_create(new_shape, 1);
        
        memcpy(new_seq->data, current->data, current->shape[0] * sizeof(float));
        new_seq->data[new_len - 1] = (float)(rand() % vocab_size);
        
        tensor_free(current);
        current = new_seq;
        
        // Simular forward pass (en producción, esto generaría el siguiente token)
        printf("  Step %d: Generado token %d\n", step + 1, (int)current->data[new_len - 1]);
    }
    
    printf("\nSecuencia final: ");
    for (int i = 0; i < current->shape[0]; i++) {
        printf("%d ", (int)current->data[i]);
    }
    printf("\n");
    
    tensor_free(current);
    transformer_free(model);
    
    printf("\n✓ Ejemplo de generación completado.\n");
}

/* ============================================================
 * EJEMPLO: ANÁLISIS DE ATENCIÓN
 * ============================================================ */

void example_attention_analysis() {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║       EJEMPLO: ANÁLISIS DE ATENCIÓN                      ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");
    
    // Demostrar cómo funciona la atención multi-cabeza
    int d_model = 64;
    int n_heads = 4;
    int seq_len = 6;
    
    printf("Demostración del mecanismo de atención:\n");
    printf("  - Dimensión del modelo: %d\n", d_model);
    printf("  - Cabezas: %d\n", n_heads);
    printf("  - Longitud de secuencia: %d\n\n", seq_len);
    
    // Crear Multi-Head Attention
    MultiHeadAttention *mha = mha_create(d_model, n_heads);
    
    // Crear entrada
    int shape[2] = {seq_len, d_model};
    Tensor *input = tensor_random(shape, 2);
    
    printf("Entrada [%dx%d] procesada por %d cabezas de atención\n\n", 
           seq_len, d_model, n_heads);
    
    // Explicar el proceso
    printf("Proceso de Multi-Head Attention:\n");
    printf("1. Proyección de Query, Key, Value\n");
    printf("   - Q = input × W_q  [%dx%d]\n", seq_len, d_model);
    printf("   - K = input × W_k  [%dx%d]\n", seq_len, d_model);
    printf("   - V = input × W_v  [%dx%d]\n", seq_len, d_model);
    printf("\n");
    printf("2. División en cabezas (cada cabeza opera en dimensión %d)\n", d_model / n_heads);
    printf("\n");
    printf("3. Cálculo de atención:\n");
    printf("   - scores = Q × K^T / sqrt(d_k)\n");
    printf("   - attention = softmax(scores) × V\n");
    printf("\n");
    printf("4. Concatenación y proyección final\n");
    
    // Ejecutar atención
    Tensor *output = mha_forward(mha, input, input, input);
    
    printf("\nSalida [%dx%d] generada exitosamente.\n", output->shape[0], output->shape[1]);
    
    // Mostrar ejemplo de pesos de atención (simplificado)
    printf("\nEjemplo de patrones de atención (conceptual):\n");
    printf("┌─────────────────────────────────────────┐\n");
    printf("│ Token 1 atiende a: Token 1, 3, 5       │\n");
    printf("│ Token 2 atiende a: Token 2, 4          │\n");
    printf("│ Token 3 atiende a: Token 1, 3, 6       │\n");
    printf("│ ...                                     │\n");
    printf("└─────────────────────────────────────────┘\n");
    
    tensor_free(input);
    tensor_free(output);
    mha_free(mha);
    
    printf("\n✓ Análisis de atención completado.\n");
}

/* ============================================================
 * EJEMPLO: COMPARACIÓN CON RED NEURONAL CLÁSICA
 * ============================================================ */

void comparison_with_classical() {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║       COMPARACIÓN: TRANSFORMER vs RED CLÁSICA            ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");
    
    printf("┌─────────────────────┬─────────────────────┬─────────────────────┐\n");
    printf("│ Característica      │ Red Clásica (RNN)   │ Transformer         │\n");
    printf("├─────────────────────┼─────────────────────┼─────────────────────┤\n");
    printf("│ Procesamiento       │ Secuencial          │ Paralelo            │\n");
    printf("│ Memoria             │ Corto plazo         │ Largo alcance       │\n");
    printf("│ Atención            │ Implícita           │ Explícita (multi-cabeza)│\n");
    printf("│ Posiciones          │ Orden importa       │ Codificación sinusoidal│\n");
    printf("│ Paralelización      │ Difícil             │ Fácil               │\n");
    printf("│ Complejidad temporal │ O(n)                │ O(n²) por capa      │\n");
    printf("└─────────────────────┴─────────────────────┴─────────────────────┘\n");
    
    printf("\nVentajas del Transformer:\n");
    printf("  1. Captura dependencias de largo alcance directamente\n");
    printf("  2. Entrenamiento altamente paralelizable\n");
    printf("  3. Mecanismo de atención interpretable\n");
    printf("  4. Base para modelos como GPT, BERT, T5\n");
    
    printf("\nDesventajas:\n");
    printf("  1. Complejidad cuadrática en longitud de secuencia\n");
    printf("  2. Requiere más datos para entrenar\n");
    printf("  3. Mayor consumo de memoria\n");
    
    printf("\nAplicaciones típicas:\n");
    printf("  - Traducción automática\n");
    printf("  - Resumen de texto\n");
    printf("  - Respuesta a preguntas\n");
    printf("  - Generación de código\n");
    printf("  - Análisis de sentimiento\n");
}

/* ============================================================
 * FUNCIÓN PRINCIPAL
 * ============================================================ */

int main() {
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║         TRANSFORMER EN C - IMPLEMENTACIÓN COMPLETA        ║\n");
    printf("║                                                          ║\n");
    printf("║  Basado en 'Attention Is All You Need' (Vaswani 2017)    ║\n");
    printf("║  Inspirado en Tensor Logic de Pedro Domingos              ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");
    
    // Inicializar semilla aleatoria
    tensor_seed(time(NULL));
    
    // Ejecutar ejemplos
    example_classification();
    
    example_text_generation();
    
    example_attention_analysis();
    
    comparison_with_classical();
    
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║                    RESUMEN FINAL                          ║\n");
    printf("╠════════════════════════════════════════════════════════════╣\n");
    printf("║  Esta implementación demuestra:                           ║\n");
    printf("║                                                           ║\n");
    printf("║  1. Operaciones tensoriales básicas en C                  ║\n");
    printf("║  2. Mecanismo de atención multi-cabeza                    ║\n");
    printf("║  3. Arquitectura Encoder-Decoder completa                 ║\n");
    printf("║  4. Positional encoding sinusoidal                        ║\n");
    printf("║  5. Layer Normalization                                   ║\n");
    printf("║  6. Feed-Forward networks con GELU                        ║\n");
    printf("║                                                           ║\n");
    printf("║  Para entrenar este modelo, necesitarías:                 ║\n");
    printf("║  - Datos de pares (source, target)                        ║\n");
    printf("║  - Función de pérdida (Cross-Entropy)                     ║\n");
    printf("║  - Optimizador (Adam con warmup)                           ║\n");
    printf("║  - Backpropagation (requiere gradientes)                  ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");
    
    return 0;
}
