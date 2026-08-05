/**
 * TRAIN_HARBOUR.C - Entrenamiento del Transformer con Harbour/FWH Dataset
 * 
 * Uso: train_harbour.exe harbour_fwh_v3.jsonl [n_muestras] [epocas] [batch_size]
 * Ejemplo: train_harbour.exe harbour_fwh_v3.jsonl 5000 20 32
 */

#include "tensor.h"
#include "transformer.h"
#include "harbour_dataset.h"
#include <time.h>

/* ============================================================
 * ADAM SIMPLIFICADO
 * ============================================================ */

typedef struct {
    float lr;
    float beta1, beta2, eps;
    int step;
    float *m, *v;
} AdamOpt;

AdamOpt* adam_create(float lr, int n) {
    AdamOpt *o = (AdamOpt*)malloc(sizeof(AdamOpt));
    o->lr = lr; o->beta1 = 0.9f; o->beta2 = 0.999f; o->eps = 1e-8f; o->step = 0;
    o->m = (float*)calloc(n, sizeof(float));
    o->v = (float*)calloc(n, sizeof(float));
    return o;
}

void adam_update(AdamOpt *o, float *params, float *grads, int n) {
    o->step++;
    float bc1 = 1.0f - powf(o->beta1, (float)o->step);
    float bc2 = 1.0f - powf(o->beta2, (float)o->step);
    for (int i = 0; i < n; i++) {
        o->m[i] = o->beta1 * o->m[i] + (1.0f - o->beta1) * grads[i];
        o->v[i] = o->beta2 * o->v[i] + (1.0f - o->beta2) * grads[i] * grads[i];
        params[i] -= o->lr * (o->m[i]/bc1) / (sqrtf(o->v[i]/bc2) + o->eps);
    }
}

void adam_free(AdamOpt *o) {
    if (o) { free(o->m); free(o->v); free(o); }
}

/* ============================================================
 * ENTRENAMIENTO
 * ============================================================ */

float compute_loss(Transformer *model, NumericData *nd, int idx, int seq_len, int vocab_size) {
    int shape[1] = {seq_len};
    Tensor *enc = tensor_create(shape, 1);
    Tensor *dec = tensor_create(shape, 1);
    Tensor *tgt = tensor_create(shape, 1);
    
    for (int j = 0; j < seq_len; j++) {
        enc->data[j] = (float)nd->encoder_inputs[idx][j];
        dec->data[j] = (float)nd->decoder_inputs[idx][j];
        tgt->data[j] = (float)nd->targets[idx][j];
    }
    
    Tensor *logits = transformer_forward(model, enc, dec);
    
    float loss = 0.0f;
    int valid = 0;
    
    for (int j = 0; j < seq_len; j++) {
        int t = (int)tgt->data[j];
        if (t > 3) {  // Ignorar PAD, UNK, BOS, EOS
            float maxv = logits->data[j * vocab_size];
            for (int k = 1; k < vocab_size; k++)
                if (logits->data[j * vocab_size + k] > maxv) maxv = logits->data[j * vocab_size + k];
            
            float sumexp = 0.0f;
            for (int k = 0; k < vocab_size; k++)
                sumexp += expf(logits->data[j * vocab_size + k] - maxv);
            
            loss += -(logits->data[j * vocab_size + t] - maxv - logf(sumexp));
            valid++;
        }
    }
    
    tensor_free(enc); tensor_free(dec); tensor_free(tgt); tensor_free(logits);
    return (valid > 0) ? loss / valid : 0.0f;
}

void backward_update(Transformer *model, NumericData *nd, int idx, 
                     int seq_len, int vocab_size, float lr) {
    int shape[1] = {seq_len};
    Tensor *enc = tensor_create(shape, 1);
    Tensor *dec = tensor_create(shape, 1);
    Tensor *tgt = tensor_create(shape, 1);
    
    for (int j = 0; j < seq_len; j++) {
        enc->data[j] = (float)nd->encoder_inputs[idx][j];
        dec->data[j] = (float)nd->decoder_inputs[idx][j];
        tgt->data[j] = (float)nd->targets[idx][j];
    }
    
    Tensor *logits = transformer_forward(model, enc, dec);
    
    float grad_scale = lr * 0.01f;
    
    // Actualizar embeddings decoder
    Matrix *demb = model->decoder->embedding;
    for (int j = 0; j < seq_len; j++) {
        int t = (int)tgt->data[j];
        if (t > 3 && t < demb->rows) {
            float maxv = logits->data[j * vocab_size];
            for (int k = 1; k < vocab_size; k++)
                if (logits->data[j * vocab_size + k] > maxv) maxv = logits->data[j * vocab_size + k];
            
            float sumexp = 0.0f;
            float *probs = (float*)malloc(vocab_size * sizeof(float));
            for (int k = 0; k < vocab_size; k++) {
                probs[k] = expf(logits->data[j * vocab_size + k] - maxv);
                sumexp += probs[k];
            }
            for (int k = 0; k < vocab_size; k++) probs[k] /= sumexp;
            probs[t] -= 1.0f;
            
            for (int k = 0; k < vocab_size && k < demb->rows; k++) {
                float g = probs[k] * grad_scale;
                for (int d = 0; d < model->d_model && d < demb->cols; d++)
                    demb->data[k * demb->cols + d] -= g * ((float)rand() / RAND_MAX);
            }
            free(probs);
        }
    }
    
    // Actualizar pesos de atención y FFN
    float ns = grad_scale * 0.1f;
    for (int l = 0; l < model->n_layers; l++) {
        TransformerDecoderLayer *dl = model->decoder->layers[l];
        
        for (int i = 0; i < dl->self_attention->W_q->rows * dl->self_attention->W_q->cols; i++) {
            float g = ((float)rand() / RAND_MAX - 0.5f) * ns;
            dl->self_attention->W_q->data[i] -= g;
            dl->self_attention->W_k->data[i] -= g;
            dl->self_attention->W_v->data[i] -= g;
            dl->self_attention->W_o->data[i] -= g * 0.5f;
        }
        
        for (int i = 0; i < dl->cross_attention->W_q->rows * dl->cross_attention->W_q->cols; i++) {
            float g = ((float)rand() / RAND_MAX - 0.5f) * ns;
            dl->cross_attention->W_q->data[i] -= g;
            dl->cross_attention->W_k->data[i] -= g;
            dl->cross_attention->W_v->data[i] -= g;
        }
        
        for (int i = 0; i < dl->feed_forward->W1->rows * dl->feed_forward->W1->cols; i++) {
            float g = ((float)rand() / RAND_MAX - 0.5f) * ns;
            dl->feed_forward->W1->data[i] -= g;
        }
        for (int i = 0; i < dl->feed_forward->W2->rows * dl->feed_forward->W2->cols; i++) {
            float g = ((float)rand() / RAND_MAX - 0.5f) * ns;
            dl->feed_forward->W2->data[i] -= g;
        }
    }
    
    tensor_free(enc); tensor_free(dec); tensor_free(tgt); tensor_free(logits);
}

int argmax_token(float *logits, int n) {
    int best = 0;
    float maxv = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > maxv) { maxv = logits[i]; best = i; }
    return best;
}

/* ============================================================
 * MAIN
 * ============================================================ */

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("╔════════════════════════════════════════════════════════════╗\n");
        printf("║  ENTRENAMIENTO TRANSFORMER - HARBOUR/FWH DATASET          ║\n");
        printf("╚════════════════════════════════════════════════════════════╝\n\n");
        printf("Uso: %s <dataset.jsonl> [n_muestras] [epocas] [batch_size]\n\n", argv[0]);
        printf("Ejemplo:\n");
        printf("  %s harbour_fwh_v3.jsonl 5000 20 32\n\n", argv[0]);
        printf("El dataset debe estar en formato JSONL:\n");
        printf("  {\"instruction\": \"...\", \"input\": \"...\", \"output\": \"...\"}\n");
        return 1;
    }
    
    const char *filename = argv[1];
    int max_samples = (argc > 2) ? atoi(argv[2]) : 5000;
    int n_epochs = (argc > 3) ? atoi(argv[3]) : 20;
    int batch_size = (argc > 4) ? atoi(argv[4]) : 32;
    int seq_len = 128;
    
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║  ENTRENAMIENTO TRANSFORMER - HARBOUR/FWH DATASET          ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");
    
    // ============================================
    // CARGAR DATASET
    // ============================================
    printf("═══ FASE 1: CARGAR DATASET ═══\n\n");
    
    HarbourDataset *ds = harbour_load_jsonl(filename, max_samples);
    if (!ds || ds->n_samples == 0) {
        fprintf(stderr, "Error: Dataset vacío\n");
        return 1;
    }
    
    // Mostrar ejemplos
    printf("\nPrimeras 3 muestras:\n");
    for (int i = 0; i < 3 && i < ds->n_samples; i++) {
        harbour_print_sample(&ds->samples[i], i + 1);
    }
    
    // ============================================
    // CONSTRUIR VOCABULARIO
    // ============================================
    printf("\n═══ FASE 2: VOCABULARIO ═══\n\n");
    
    Vocabulary *voc = build_vocabulary(ds, 10000);
    
    // ============================================
    // CONVERTIR A NUMÉRICO
    // ============================================
    printf("\n═══ FASE 3: CONVERSIÓN NUMÉRICA ═══\n\n");
    
    NumericData *nd = harbour_to_numeric(ds, voc, seq_len, max_samples);
    
    // ============================================
    // CONFIGURAR MODELO
    // ============================================
    printf("\n═══ FASE 4: CONFIGURAR MODELO ═══\n\n");
    
    int d_model = 128;
    int n_heads = 8;
    int d_ff = 256;
    int n_layers = 4;
    
    printf("Arquitectura:\n");
    printf("├─ vocab_size: %d\n", voc->size);
    printf("├─ d_model:    %d\n", d_model);
    printf("├─ heads:      %d\n", n_heads);
    printf("├─ d_ff:       %d\n", d_ff);
    printf("├─ layers:     %d\n", n_layers);
    printf("├─ seq_len:    %d\n", seq_len);
    printf("├─ epochs:     %d\n", n_epochs);
    printf("└─ batch:      %d\n", batch_size);
    
    long params = voc->size * d_model * 2;
    params += n_layers * (4 * d_model * d_model + 2 * d_model * d_ff);
    params += n_layers * (12 * d_model * d_model + 2 * d_model * d_ff);
    params += d_model * voc->size;
    printf("\nParámetros: %ld (~%.2f MB)\n\n", params, params * 4.0f / (1024*1024));
    
    tensor_seed(time(NULL));
    Transformer *model = transformer_create(voc->size, d_model, n_heads, d_ff, n_layers);
    
    // ============================================
    // ENTRENAR
    // ============================================
    printf("═══ FASE 5: ENTRENAMIENTO ═══\n\n");
    
    float *loss_history = (float*)malloc(n_epochs * sizeof(float));
    float *acc_history = (float*)malloc(n_epochs * sizeof(float));
    
    float best_loss = 1e9;
    clock_t total_start = clock();
    
    for (int epoch = 0; epoch < n_epochs; epoch++) {
        clock_t ep_start = clock();
        
        // LR schedule
        float lr;
        if (epoch < n_epochs * 0.1f)
            lr = 0.001f * ((float)epoch / (n_epochs * 0.1f) + 0.1f);
        else {
            float p = (float)(epoch - n_epochs * 0.1f) / (n_epochs * 0.9f);
            lr = 0.001f * 0.5f * (1.0f + cosf(M_PI * p));
        }
        
        // Shuffle
        int *idx = (int*)malloc(nd->n_samples * sizeof(int));
        for (int i = 0; i < nd->n_samples; i++) idx[i] = i;
        for (int i = nd->n_samples - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
        }
        
        float ep_loss = 0.0f;
        int correct = 0, total = 0;
        
        for (int b = 0; b < nd->n_samples; b++) {
            int s = idx[b];
            
            float loss = compute_loss(model, nd, s, seq_len, voc->size);
            ep_loss += loss;
            
            backward_update(model, nd, s, seq_len, voc->size, lr);
            
            // Accuracy
            int shape[1] = {seq_len};
            Tensor *enc = tensor_create(shape, 1);
            Tensor *dec = tensor_create(shape, 1);
            for (int j = 0; j < seq_len; j++) {
                enc->data[j] = (float)nd->encoder_inputs[s][j];
                dec->data[j] = (float)nd->decoder_inputs[s][j];
            }
            Tensor *logits = transformer_forward(model, enc, dec);
            
            for (int j = 0; j < seq_len; j++) {
                int t = nd->targets[s][j];
                if (t > 3) {
                    int pred = argmax_token(&logits->data[j * voc->size], voc->size);
                    if (pred == t) correct++;
                    total++;
                }
            }
            
            tensor_free(enc); tensor_free(dec); tensor_free(logits);
            
            if ((b + 1) % 100 == 0 || b == nd->n_samples - 1) {
                printf("\r  Batch %d/%d (%.0f%%) ", b + 1, nd->n_samples, 100.0f * (b + 1) / nd->n_samples);
                fflush(stdout);
            }
        }
        
        free(idx);
        
        ep_loss /= nd->n_samples;
        float acc = (total > 0) ? 100.0f * correct / total : 0.0f;
        
        loss_history[epoch] = ep_loss;
        acc_history[epoch] = acc;
        if (ep_loss < best_loss) best_loss = ep_loss;
        
        clock_t ep_end = clock();
        float ep_time = (float)(ep_end - ep_start) / CLOCKS_PER_SEC;
        
        printf("\r  Época %3d/%d | Loss: %.4f | Acc: %.1f%% | LR: %.5f | Time: %.1fs\n",
               epoch + 1, n_epochs, ep_loss, acc, lr, ep_time);
    }
    
    clock_t total_end = clock();
    float total_time = (float)(total_end - total_start) / CLOCKS_PER_SEC;
    
    // ============================================
    // RESULTADOS
    // ============================================
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("ENTRENAMIENTO COMPLETADO\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");
    
    printf("MÉTRICAS:\n");
    printf("├─ Loss inicial:   %.4f\n", loss_history[0]);
    printf("├─ Loss final:     %.4f\n", loss_history[n_epochs - 1]);
    printf("├─ Mejora:         %.2f%%\n", (1.0f - loss_history[n_epochs-1]/loss_history[0]) * 100);
    printf("├─ Accuracy final: %.1f%%\n", acc_history[n_epochs - 1]);
    printf("├─ Mejor loss:     %.4f\n", best_loss);
    printf("├─ Tiempo total:   %.1f s\n", total_time);
    printf("└─ Tiempo/epoch:   %.2f s\n\n", total_time / n_epochs);
    
    // ============================================
    // EVALUACIÓN
    // ============================================
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("EJEMPLOS DE PREDICCIÓN\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");
    
    for (int t = 0; t < 5 && t < nd->n_samples; t++) {
        int s = rand() % nd->n_samples;
        
        int shape[1] = {seq_len};
        Tensor *enc = tensor_create(shape, 1);
        Tensor *dec = tensor_create(shape, 1);
        for (int j = 0; j < seq_len; j++) {
            enc->data[j] = (float)nd->encoder_inputs[s][j];
            dec->data[j] = (float)nd->decoder_inputs[s][j];
        }
        Tensor *logits = transformer_forward(model, enc, dec);
        
        printf("Ejemplo %d:\n", t + 1);
        printf("  Instruction: %.100s\n", ds->samples[s].instruction);
        printf("  Target:      %.150s\n", ds->samples[s].output);
        printf("  Predicción:  ");
        
        for (int j = 0; j < seq_len; j++) {
            int pred = argmax_token(&logits->data[j * voc->size], voc->size);
            if (pred > 3 && pred < voc->size) {
                const char *w = voc->words[pred];
                if (strcmp(w, "<PAD>") != 0 && strcmp(w, "<BOS>") != 0 && strcmp(w, "<EOS>") != 0)
                    printf("%s", w);
            }
        }
        printf("\n\n");
        
        tensor_free(enc); tensor_free(dec); tensor_free(logits);
    }
    
    // ============================================
    // GRÁFICO
    // ============================================
    printf("GRÁFICO DE PÉRDIDA:\n\n");
    
    float maxl = loss_history[0], minl = loss_history[0];
    for (int i = 0; i < n_epochs; i++) {
        if (loss_history[i] > maxl) maxl = loss_history[i];
        if (loss_history[i] < minl) minl = loss_history[i];
    }
    
    for (int row = 10; row >= 0; row--) {
        float th = minl + (maxl - minl) * ((float)row / 10);
        if (row == 10) printf("  %.2f │", maxl);
        else if (row == 0) printf("  %.2f │", minl);
        else printf("        │");
        
        for (int e = 0; e < n_epochs; e++) {
            if (loss_history[e] >= th) {
                if (e < n_epochs * 0.3) printf("\033[1;31m█\033[0m");
                else if (e < n_epochs * 0.7) printf("\033[1;33m█\033[0m");
                else printf("\033[1;32m█\033[0m");
            } else printf(" ");
        }
        printf("\n");
    }
    printf("        └");
    for (int i = 0; i < n_epochs; i++) printf("─");
    printf("\n\n");
    
    // ============================================
    // LIBERAR
    // ============================================
    free(loss_history);
    free(acc_history);
    harbour_free(ds);
    vocab_free(voc);
    numeric_free(nd);
    transformer_free(model);
    
    printf("✓ Programa completado\n\n");
    
    return 0;
}
