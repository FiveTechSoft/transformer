/**
 * HARBOUR_DATASET.C - Parser JSONL para Harbour/FWH Dataset
 * 
 * Lee el formato JSONL del dataset de fivetech:
 * {"system": "...", "instruction": "...", "input": "...", "output": "...", "task_type": "..."}
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>

/* ============================================================
 * ESTRUCTURAS
 * ============================================================ */

typedef struct {
    char *system;
    char *instruction;
    char *input;
    char *output;
    char *task_type;
} HarbourSample;

typedef struct {
    HarbourSample *samples;
    int n_samples;
    int max_samples;
} HarbourDataset;

typedef struct {
    char **tokens;
    int *ids;
    int n_tokens;
    int max_id;
} TokenList;

typedef struct {
    char **words;
    int *ids;
    int size;
    int capacity;
    int unk_id;
    int bos_id;
    int eos_id;
    int pad_id;
} Vocabulary;

/* ============================================================
 * PARSER JSONL
 * ============================================================ */

/**
 * Extraer valor de campo JSON simple
 * Busca: "campo": "valor"
 */
char* json_get_field(const char *json, const char *field) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", field);
    
    char *start = strstr(json, search);
    if (!start) return NULL;
    
    // Buscar ':'
    start = strchr(start + strlen(search), ':');
    if (!start) return NULL;
    start++;
    
    // Saltar espacios
    while (*start == ' ' || *start == '\t') start++;
    
    // Buscar comilla de apertura
    if (*start != '"') return NULL;
    start++;
    
    // Buscar comilla de cierre (escapando \" dentro)
    char *end = start;
    while (*end) {
        if (*end == '"' && *(end-1) != '\\') break;
        end++;
    }
    
    int len = end - start;
    char *value = (char*)malloc(len + 1);
    strncpy(value, start, len);
    value[len] = '\0';
    
    return value;
}

/**
 * Cargar dataset JSONL
 */
HarbourDataset* harbour_load_jsonl(const char *filename, int max_samples) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: No se pudo abrir %s\n", filename);
        return NULL;
    }
    
    HarbourDataset *ds = (HarbourDataset*)malloc(sizeof(HarbourDataset));
    ds->max_samples = max_samples;
    ds->samples = (HarbourSample*)malloc(max_samples * sizeof(HarbourSample));
    ds->n_samples = 0;
    
    char line[65536];
    
    while (fgets(line, sizeof(line), file) && ds->n_samples < max_samples) {
        // Ignorar líneas vacías
        if (line[0] == '\n' || line[0] == '\r') continue;
        
        HarbourSample *s = &ds->samples[ds->n_samples];
        
        s->system = json_get_field(line, "system");
        s->instruction = json_get_field(line, "instruction");
        s->input = json_get_field(line, "input");
        s->output = json_get_field(line, "output");
        s->task_type = json_get_field(line, "task_type");
        
        // Si no hay system, crear uno vacío
        if (!s->system) s->system = strdup("");
        if (!s->instruction) s->instruction = strdup("");
        if (!s->input) s->input = strdup("");
        if (!s->output) s->output = strdup("");
        if (!s->task_type) s->task_type = strdup("unknown");
        
        ds->n_samples++;
    }
    
    fclose(file);
    
    printf("Dataset cargado: %d muestras de %s\n", ds->n_samples, filename);
    return ds;
}

/**
 * Liberar dataset
 */
void harbour_free(HarbourDataset *ds) {
    if (ds) {
        for (int i = 0; i < ds->n_samples; i++) {
            free(ds->samples[i].system);
            free(ds->samples[i].instruction);
            free(ds->samples[i].input);
            free(ds->samples[i].output);
            free(ds->samples[i].task_type);
        }
        free(ds->samples);
        free(ds);
    }
}

/**
 * Imprimir muestra
 */
void harbour_print_sample(HarbourSample *s, int idx) {
    printf("\n=== Muestra %d ===\n", idx);
    printf("Task: %s\n", s->task_type);
    printf("System: %.100s...\n", s->system);
    printf("Instruction: %.100s...\n", s->instruction);
    if (strlen(s->input) > 0) {
        printf("Input: %.100s...\n", s->input);
    }
    printf("Output: %.200s...\n", s->output);
}

/* ============================================================
 * TOKENIZACIÓN SIMPLE
 * ============================================================ */

/**
 * Tokenizar texto en palabras/caracteres
 * Retorna lista de tokens
 */
TokenList* tokenize_text(const char *text) {
    TokenList *tl = (TokenList*)malloc(sizeof(TokenList));
    tl->tokens = (char**)malloc(1024 * sizeof(char*));
    tl->ids = (int*)malloc(1024 * sizeof(int));
    tl->n_tokens = 0;
    tl->max_id = 0;
    
    int len = strlen(text);
    int i = 0;
    
    while (i < len && tl->n_tokens < 1024) {
        // Saltar espacios
        while (i < len && (text[i] == ' ' || text[i] == '\t')) i++;
        if (i >= len) break;
        
        // Detectar si es carácter especial
        if (text[i] == '\n' || text[i] == '\r' || text[i] == '\t') {
            tl->tokens[tl->n_tokens] = (char*)malloc(2);
            tl->tokens[tl->n_tokens][0] = text[i];
            tl->tokens[tl->n_tokens][1] = '\0';
            tl->n_tokens++;
            i++;
        }
        // Detectar si es número
        else if (isdigit(text[i])) {
            int start = i;
            while (i < len && isdigit(text[i])) i++;
            int num_len = i - start;
            tl->tokens[tl->n_tokens] = (char*)malloc(num_len + 1);
            strncpy(tl->tokens[tl->n_tokens], text + start, num_len);
            tl->tokens[tl->n_tokens][num_len] = '\0';
            tl->n_tokens++;
        }
        // Detectar si es letra
        else if (isalpha(text[i]) || text[i] == '_') {
            int start = i;
            while (i < len && (isalnum(text[i]) || text[i] == '_')) i++;
            int word_len = i - start;
            tl->tokens[tl->n_tokens] = (char*)malloc(word_len + 1);
            strncpy(tl->tokens[tl->n_tokens], text + start, word_len);
            tl->tokens[tl->n_tokens][word_len] = '\0';
            tl->n_tokens++;
        }
        // Carácter especial (operadores, etc.)
        else {
            tl->tokens[tl->n_tokens] = (char*)malloc(2);
            tl->tokens[tl->n_tokens][0] = text[i];
            tl->tokens[tl->n_tokens][1] = '\0';
            tl->n_tokens++;
            i++;
        }
    }
    
    return tl;
}

/**
 * Liberar lista de tokens
 */
void free_token_list(TokenList *tl) {
    if (tl) {
        for (int i = 0; i < tl->n_tokens; i++) {
            free(tl->tokens[i]);
        }
        free(tl->tokens);
        free(tl->ids);
        free(tl);
    }
}

/* ============================================================
 * VOCABULARIO
 * ============================================================ */

Vocabulary* vocab_create(int capacity) {
    Vocabulary *v = (Vocabulary*)malloc(sizeof(Vocabulary));
    v->words = (char**)malloc(capacity * sizeof(char*));
    v->ids = (int*)malloc(capacity * sizeof(int));
    v->size = 0;
    v->capacity = capacity;
    
    // Tokens especiales
    v->words[0] = strdup("<PAD>");
    v->words[1] = strdup("<UNK>");
    v->words[2] = strdup("<BOS>");
    v->words[3] = strdup("<EOS>");
    v->size = 4;
    
    v->pad_id = 0;
    v->unk_id = 1;
    v->bos_id = 2;
    v->eos_id = 3;
    
    return v;
}

int vocab_add(Vocabulary *v, const char *word) {
    // Buscar si ya existe
    for (int i = 4; i < v->size; i++) {
        if (strcmp(v->words[i], word) == 0) {
            return i;
        }
    }
    
    // Agregar nuevo
    if (v->size >= v->capacity) {
        v->capacity *= 2;
        v->words = (char**)realloc(v->words, v->capacity * sizeof(char*));
    }
    
    v->words[v->size] = strdup(word);
    v->size++;
    return v->size - 1;
}

int vocab_lookup(Vocabulary *v, const char *word) {
    for (int i = 0; i < v->size; i++) {
        if (strcmp(v->words[i], word) == 0) {
            return i;
        }
    }
    return v->unk_id;
}

void vocab_free(Vocabulary *v) {
    if (v) {
        for (int i = 0; i < v->size; i++) {
            free(v->words[i]);
        }
        free(v->words);
        free(v->ids);
        free(v);
    }
}

/* ============================================================
 * CONSTRUIR VOCABULARIO DEL DATASET
 * ============================================================ */

Vocabulary* build_vocabulary(HarbourDataset *ds, int max_vocab) {
    printf("Construyendo vocabulario...\n");
    
    Vocabulary *v = vocab_create(max_vocab);
    
    for (int i = 0; i < ds->n_samples; i++) {
        // Tokenizar y agregar palabras del output
        TokenList *tl = tokenize_text(ds->samples[i].output);
        for (int j = 0; j < tl->n_tokens; j++) {
            vocab_add(v, tl->tokens[j]);
        }
        free_token_list(tl);
        
        // Tokenizar instruction
        tl = tokenize_text(ds->samples[i].instruction);
        for (int j = 0; j < tl->n_tokens; j++) {
            vocab_add(v, tl->tokens[j]);
        }
        free_token_list(tl);
    }
    
    printf("Vocabulario: %d tokens\n", v->size);
    return v;
}

/* ============================================================
 * EXPORTAR A FORMATO NUMÉRICO PARA TRAINING
 * ============================================================ */

typedef struct {
    int **encoder_inputs;
    int **decoder_inputs;
    int **targets;
    int n_samples;
    int seq_len;
    int vocab_size;
} NumericData;

/**
 * Convertir dataset a secuencias numéricas
 */
NumericData* harbour_to_numeric(HarbourDataset *ds, Vocabulary *voc, int seq_len, int max_samples) {
    int n = (ds->n_samples < max_samples) ? ds->n_samples : max_samples;
    
    NumericData *nd = (NumericData*)malloc(sizeof(NumericData));
    nd->n_samples = n;
    nd->seq_len = seq_len;
    nd->vocab_size = voc->size;
    
    nd->encoder_inputs = (int**)malloc(n * sizeof(int*));
    nd->decoder_inputs = (int**)malloc(n * sizeof(int*));
    nd->targets = (int**)malloc(n * sizeof(int*));
    
    for (int i = 0; i < n; i++) {
        nd->encoder_inputs[i] = (int*)calloc(seq_len, sizeof(int));
        nd->decoder_inputs[i] = (int*)calloc(seq_len, sizeof(int));
        nd->targets[i] = (int*)calloc(seq_len, sizeof(int));
        
        // Crear prompt: instruction + input
        char prompt[4096];
        snprintf(prompt, sizeof(prompt), "%s %s", 
                ds->samples[i].instruction,
                ds->samples[i].input);
        
        // Tokenizar prompt (encoder)
        TokenList *enc_tokens = tokenize_text(prompt);
        int idx = 0;
        nd->encoder_inputs[i][idx++] = voc->bos_id;
        for (int j = 0; j < enc_tokens->n_tokens && idx < seq_len - 2; j++) {
            nd->encoder_inputs[i][idx++] = vocab_lookup(voc, enc_tokens->tokens[j]);
        }
        nd->encoder_inputs[i][idx++] = voc->eos_id;
        free_token_list(enc_tokens);
        
        // Tokenizar output (decoder + target)
        TokenList *dec_tokens = tokenize_text(ds->samples[i].output);
        idx = 0;
        nd->decoder_inputs[i][idx++] = voc->bos_id;
        for (int j = 0; j < dec_tokens->n_tokens && idx < seq_len - 1; j++) {
            int token_id = vocab_lookup(voc, dec_tokens->tokens[j]);
            nd->decoder_inputs[i][idx] = token_id;
            if (idx > 0) {
                nd->targets[i][idx - 1] = token_id;
            }
            idx++;
        }
        if (idx < seq_len) {
            nd->targets[i][idx - 1] = voc->eos_id;
        }
        nd->decoder_inputs[i][idx] = voc->eos_id;
        
        free_token_list(dec_tokens);
    }
    
    printf("Datos numéricos: %d muestras, seq_len=%d, vocab=%d\n", n, seq_len, voc->size);
    return nd;
}

void numeric_free(NumericData *nd) {
    if (nd) {
        for (int i = 0; i < nd->n_samples; i++) {
            free(nd->encoder_inputs[i]);
            free(nd->decoder_inputs[i]);
            free(nd->targets[i]);
        }
        free(nd->encoder_inputs);
        free(nd->decoder_inputs);
        free(nd->targets);
        free(nd);
    }
}
