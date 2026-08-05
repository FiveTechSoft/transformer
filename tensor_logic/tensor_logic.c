/**
 * TENSOR_LOGIC.C - Implementación del Tensor Logic
 * 
 * Este archivo implementa los principios del Tensor Logic
 * de Pedro Domingos para unificar lógica y aprendizaje.
 * 
 * Principios clave:
 * 1. Los pesos de las cláusulas se aprenden por gradientes
 * 2. La inferencia se realiza via operaciones tensoriales
 * 3. La incertidumbre se modela con valores de verdad difusos
 */

#include "tensor_logic.h"

/* ============================================================
 * FUNCIONES DE CREACIÓN
 * ============================================================ */

Literal* literal_create(const char *name, int arity, float weight) {
    Literal *lit = (Literal*)malloc(sizeof(Literal));
    if (!lit) return NULL;
    
    lit->name = strdup(name);
    lit->arity = arity;
    lit->weight = weight;
    lit->truth_value = 0.5f;  // Valor por defecto (incertidumbre)
    
    // Crear embedding para el literal
    int shape[1] = {arity + 1};  // +1 para el peso
    lit->embedding = tensor_create(shape, 1);
    
    // Inicializar embedding con el peso y arity
    lit->embedding->data[0] = weight;
    for (int i = 1; i <= arity; i++) {
        lit->embedding->data[i] = (float)arity / 10.0f;
    }
    
    return lit;
}

Clause* clause_create(int n_literals) {
    Clause *clause = (Clause*)malloc(sizeof(Clause));
    if (!clause) return NULL;
    
    clause->n_literals = n_literals;
    clause->literals = (Literal**)malloc(n_literals * sizeof(Literal*));
    clause->weight = 1.0f;
    clause->satisfaction = 0.0f;
    
    // Crear tensor de relación
    int shape[2] = {n_literals, n_literals};
    clause->relation_tensor = tensor_zeros(shape, 2);
    
    return clause;
}

TensorLogicLayer* tensor_logic_layer_create(int input_dim, int logic_dim, 
                                             int n_predicates, TNormType tnorm) {
    TensorLogicLayer *layer = (TensorLogicLayer*)malloc(sizeof(TensorLogicLayer));
    if (!layer) return NULL;
    
    layer->input_dim = input_dim;
    layer->logic_dim = logic_dim;
    layer->n_predicates = n_predicates;
    layer->tnorm = tnorm;
    
    // Proyección al espacio lógico
    layer->W_logic = matrix_create(input_dim, logic_dim);
    
    // Pesos para operaciones lógicas
    int shape[1] = {logic_dim};
    layer->and_weights = tensor_create(shape, 1);
    layer->or_weights = tensor_create(shape, 1);
    layer->not_weights = tensor_create(shape, 1);
    layer->implies_weights = tensor_create(shape, 1);
    
    // Inicializar pesos de operaciones lógicas
    for (int i = 0; i < logic_dim; i++) {
        layer->and_weights->data[i] = 1.0f;
        layer->or_weights->data[i] = 1.0f;
        layer->not_weights->data[i] = -1.0f;
        layer->implies_weights->data[i] = 1.0f;
    }
    
    // Embeddings de predicados
    layer->predicate_embeddings = matrix_create(n_predicates, logic_dim);
    
    // Inicialización Xavier
    float limit = sqrt(6.0f / (float)(input_dim + logic_dim));
    for (int i = 0; i < input_dim * logic_dim; i++) {
        layer->W_logic->data[i] = ((float)rand() / RAND_MAX) * 2.0f * limit - limit;
    }
    
    for (int i = 0; i < n_predicates * logic_dim; i++) {
        layer->predicate_embeddings->data[i] = ((float)rand() / RAND_MAX) * 2.0f * limit - limit;
    }
    
    return layer;
}

MarkovLogicNetwork* markov_logic_create(int n_clauses, int n_variables, 
                                        int n_predicates, TNormType tnorm) {
    MarkovLogicNetwork *mln = (MarkovLogicNetwork*)malloc(sizeof(MarkovLogicNetwork));
    if (!mln) return NULL;
    
    mln->n_clauses = n_clauses;
    mln->n_variables = n_variables;
    mln->n_predicates = n_predicates;
    mln->tnorm = tnorm;
    
    // Crear cláusulas
    mln->clauses = (Clause**)malloc(n_clauses * sizeof(Clause*));
    for (int i = 0; i < n_clauses; i++) {
        int n_literals = 2 + rand() % 3;  // 2-4 literales por cláusula
        mln->clauses[i] = clause_create(n_literals);
        mln->clauses[i]->weight = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    // Crear tensores
    int adj_shape[2] = {n_variables, n_variables};
    mln->adjacency = tensor_zeros(adj_shape, 2);
    
    int weight_shape[1] = {n_clauses};
    mln->clause_weights = tensor_create(weight_shape, 1);
    for (int i = 0; i < n_clauses; i++) {
        mln->clause_weights->data[i] = mln->clauses[i]->weight;
    }
    
    int truth_shape[1] = {n_variables};
    mln->truth_values = tensor_create(truth_shape, 1);
    for (int i = 0; i < n_variables; i++) {
        mln->truth_values->data[i] = 0.5f;  // Incertidumbre inicial
    }
    
    return mln;
}

FuzzyTransformerLayer* fuzzy_transformer_create(int d_model, int n_predicates,
                                                 TNormType tnorm) {
    FuzzyTransformerLayer *layer = (FuzzyTransformerLayer*)malloc(sizeof(FuzzyTransformerLayer));
    if (!layer) return NULL;
    
    // Capa de lógica
    layer->logic_layer = tensor_logic_layer_create(d_model, d_model, n_predicates, tnorm);
    
    // Pesos para atención lógica
    layer->W_q_logic = matrix_create(d_model, d_model);
    layer->W_k_logic = matrix_create(d_model, d_model);
    layer->W_v_logic = matrix_create(d_model, d_model);
    
    // Inicializar pesos
    float limit = sqrt(6.0f / (float)d_model);
    for (int i = 0; i < d_model * d_model; i++) {
        layer->W_q_logic->data[i] = ((float)rand() / RAND_MAX) * 2.0f * limit - limit;
        layer->W_k_logic->data[i] = ((float)rand() / RAND_MAX) * 2.0f * limit - limit;
        layer->W_v_logic->data[i] = ((float)rand() / RAND_MAX) * 2.0f * limit - limit;
    }
    
    // Base de conocimiento (vacía inicialmente)
    int kb_shape[2] = {n_predicates, d_model};
    layer->knowledge_base = tensor_zeros(kb_shape, 2);
    
    // Operadores difusos por defecto
    layer->union_op = fuzzy_or;
    layer->intersection_op = NULL;  // Se usará fuzzy_and directamente
    
    return layer;
}

/* ============================================================
 * OPERACIONES LÓGICAS DIFUSAS
 * ============================================================ */

/**
 * AND lógico difuso (intersección / t-norma)
 * 
 * Diferentes t-normas:
 * - Producto: a ⊗ b = a * b
 * - Łukasiewicz: a ⊗ b = max(0, a + b - 1)
 * - Gödel: a ⊗ b = min(a, b)
 */
float fuzzy_and(float a, float b, TNormType tnorm) {
    switch (tnorm) {
        case TNORM_PRODUCT:
            return a * b;
        case TNORM_LUKASIEWICZ:
            return fmaxf(0.0f, a + b - 1.0f);
        case TNORM_GODEL:
            return fminf(a, b);
        default:
            return a * b;
    }
}

/**
 * OR lógico difuso (unión / t-conorma)
 * 
 * Co-norma dual de la t-norma:
 * - Suma probabilística: a ⊕ b = a + b - a*b
 * - Łukasiewicz: a ⊕ b = min(1, a + b)
 * - Gödel: a ⊕ b = max(a, b)
 */
float fuzzy_or(float a, float b) {
    // Suma probabilística (dual del producto)
    return a + b - a * b;
}

/**
 * NOT lógico difuso (complemento)
 * 
 * Complemento simple: ¬a = 1 - a
 */
float fuzzy_not(float a) {
    return 1.0f - a;
}

/**
 * IMPLIES lógico difuso
 * 
 * a → b = ¬a ⊕ b = (1 - a) + b - (1 - a)*b
 * O simplificado: min(1, 1 - a + b)
 */
float fuzzy_implies(float a, float b) {
    return fminf(1.0f, 1.0f - a + b);
}

/**
 * Aplicar t-norma a tensor (elemento a elemento)
 */
Tensor* apply_tnorm(Tensor *a, Tensor *b, TNormType tnorm) {
    if (a->size != b->size) {
        fprintf(stderr, "Error: Tamaños incompatibles para t-norma\n");
        return NULL;
    }
    
    Tensor *result = tensor_copy(a);
    if (!result) return NULL;
    
    for (int i = 0; i < result->size; i++) {
        result->data[i] = fuzzy_and(a->data[i], b->data[i], tnorm);
    }
    
    return result;
}

/* ============================================================
 * RAZONAMIENTO LÓGICO
 * ============================================================ */

/**
 * Inferencia lógica en Markov Logic Network
 * 
 * Calcula la probabilidad de cada variable dado:
 * 1. Las cláusulas (reglas)
 * 2. Los pesos de las cláusulas
 * 3. La evidencia observada
 */
Tensor* logical_inference(MarkovLogicNetwork *mln, Tensor *evidence) {
    // Copiar evidencia como valores iniciales
    tensor_free(mln->truth_values);
    mln->truth_values = tensor_copy(evidence);
    
    // Iterar hasta convergencia (belief propagation simplificado)
    int max_iter = 10;
    float convergence_threshold = 0.01f;
    
    for (int iter = 0; iter < max_iter; iter++) {
        float max_change = 0.0f;
        
        // Para cada cláusula
        for (int c = 0; c < mln->n_clauses; c++) {
            Clause *clause = mln->clauses[c];
            
            // Calcular satisfacción de la cláusula
            float satisfaction = 1.0f;
            for (int l = 0; l < clause->n_literals; l++) {
                // Obtener valor de verdad del literal
                int var_idx = l % mln->n_variables;
                float truth = mln->truth_values->data[var_idx];
                
                // Aplicar t-norma
                satisfaction = fuzzy_and(satisfaction, truth, mln->tnorm);
            }
            
            clause->satisfaction = satisfaction;
            
            // Propagar evidencia a las variables
            float weight = mln->clause_weights->data[c];
            
            for (int l = 0; l < clause->n_literals; l++) {
                int var_idx = l % mln->n_variables;
                
                // Calcular influencia de la cláusula en la variable
                float influence = weight * (satisfaction - 0.5f);
                
                // Actualizar valor de verdad
                float old_value = mln->truth_values->data[var_idx];
                float new_value = old_value + 0.1f * influence;
                new_value = fmaxf(0.0f, fminf(1.0f, new_value));  // Clamp [0, 1]
                
                mln->truth_values->data[var_idx] = new_value;
                
                max_change = fmaxf(max_change, fabsf(new_value - old_value));
            }
        }
        
        // Verificar convergencia
        if (max_change < convergence_threshold) {
            break;
        }
    }
    
    return tensor_copy(mln->truth_values);
}

/**
 * Aplicar reglas lógicas difusas
 * 
 * Transforma la entrada aplicando operaciones lógicas
 * con los predicados de la capa.
 */
Tensor* apply_rules(FuzzyTransformerLayer *layer, Tensor *input) {
    int seq_len = input->shape[0];
    int d_model = input->shape[1];
    int n_predicates = layer->logic_layer->n_predicates;
    
    // Proyectar al espacio lógico
    Matrix *X = matrix_create_from_data(input->data, seq_len, d_model);
    Matrix *logic_space = matrix_multiply(X, layer->logic_layer->W_logic);
    
    // Aplicar operaciones lógicas con cada predicado
    Tensor *result = tensor_create(input->shape, input->ndim);
    
    for (int p = 0; p < n_predicates; p++) {
        // Obtener embedding del predicado
        float *pred_emb = &layer->logic_layer->predicate_embeddings->data[p * d_model];
        
        // Calcular similitud con cada posición
        for (int i = 0; i < seq_len; i++) {
            float similarity = 0.0f;
            for (int j = 0; j < d_model; j++) {
                similarity += logic_space->data[i * d_model + j] * pred_emb[j];
            }
            similarity = tanhf(similarity);  // Normalizar a [-1, 1]
            
            // Convertir a valor de verdad [0, 1]
            float truth_value = (similarity + 1.0f) / 2.0f;
            
            // Aplicar operaciones lógicas
            float and_result = fuzzy_and(truth_value, 0.8f, layer->logic_layer->tnorm);
            float or_result = fuzzy_or(truth_value, 0.2f);
            float not_result = fuzzy_not(truth_value);
            float implies_result = fuzzy_implies(truth_value, 0.5f);
            
            // Combinar resultados con pesos aprendibles
            float combined = 0.0f;
            combined += and_result * layer->logic_layer->and_weights->data[0];
            combined += or_result * layer->logic_layer->or_weights->data[0];
            combined += not_result * layer->logic_layer->not_weights->data[0];
            combined += implies_result * layer->logic_layer->implies_weights->data[0];
            
            // Agregar al resultado
            result->data[i * d_model + p % d_model] += combined / n_predicates;
        }
    }
    
    matrix_free(X);
    matrix_free(logic_space);
    
    return result;
}

/**
 * Verificar consistencia lógica de la MLN
 * 
 * Retorna un valor entre 0 (completamente inconsistente)
 * y 1 (completamente consistente).
 */
float check_consistency(MarkovLogicNetwork *mln) {
    float total_consistency = 0.0f;
    
    for (int c = 0; c < mln->n_clauses; c++) {
        Clause *clause = mln->clauses[c];
        
        // Verificar si la cláusula es consistente
        float consistency = 1.0f;
        
        for (int l = 0; l < clause->n_literals - 1; l++) {
            int var1 = l % mln->n_variables;
            int var2 = (l + 1) % mln->n_variables;
            
            float truth1 = mln->truth_values->data[var1];
            float truth2 = mln->truth_values->data[var2];
            
            // Verificar implicación
            float implication = fuzzy_implies(truth1, truth2);
            consistency = fuzzy_and(consistency, implication, mln->tnorm);
        }
        
        total_consistency += consistency * fabsf(clause->weight);
    }
    
    return total_consistency / mln->n_clauses;
}

/* ============================================================
 * ATENCIÓN LÓGICA
 * ============================================================ */

/**
 * Score de atención difuso
 * 
 * Calcula la atención usando t-norma en lugar de producto punto.
 * Esto permite que la atención capture relaciones lógicas.
 */
Tensor* fuzzy_attention_score(Tensor *q, Tensor *k, TNormType tnorm) {
    int q_len = q->shape[0];
    int k_len = k->shape[0];
    int d = q->shape[1];
    
    int shape[2] = {q_len, k_len};
    Tensor *scores = tensor_create(shape, 2);
    
    for (int i = 0; i < q_len; i++) {
        for (int j = 0; j < k_len; j++) {
            // Calcular "similitud lógica"
            float score = 0.0f;
            
            for (int d_idx = 0; d_idx < d; d_idx++) {
                float q_val = q->data[i * d + d_idx];
                float k_val = k->data[j * d + d_idx];
                
                // Convertir a valores de verdad [0, 1]
                float q_truth = (q_val + 1.0f) / 2.0f;
                float k_truth = (k_val + 1.0f) / 2.0f;
                
                // Aplicar t-norma (similitud lógica)
                score += fuzzy_and(q_truth, k_truth, tnorm);
            }
            
            scores->data[i * k_len + j] = score / d;
        }
    }
    
    return scores;
}

/**
 * Multi-Head Attention con operaciones lógicas
 * 
 * Combina atención estándar con razonamiento lógico
 */
Tensor* logical_attention(FuzzyTransformerLayer *layer, 
                          Tensor *q, Tensor *k, Tensor *v) {
    int seq_len = q->shape[0];
    int d_model = q->shape[1];
    TNormType tnorm = layer->logic_layer->tnorm;
    
    // Proyecciones lineales
    Matrix *Q = matrix_create_from_data(q->data, seq_len, d_model);
    Matrix *Q_proj = matrix_multiply(Q, layer->W_q_logic);
    
    Matrix *K = matrix_create_from_data(k->data, seq_len, d_model);
    Matrix *K_proj = matrix_multiply(K, layer->W_k_logic);
    
    Matrix *V_mat = matrix_create_from_data(v->data, seq_len, d_model);
    Matrix *V_proj = matrix_multiply(V_mat, layer->W_v_logic);
    
    // Calcular scores con t-norma
    Tensor *Q_tensor = tensor_create_from_data(Q_proj->data, q->shape, q->ndim);
    Tensor *K_tensor = tensor_create_from_data(K_proj->data, k->shape, k->ndim);
    
    Tensor *scores = fuzzy_attention_score(Q_tensor, K_tensor, tnorm);
    
    // Aplicar softmax (para normalizar)
    for (int i = 0; i < seq_len; i++) {
        float max_val = scores->data[i * seq_len];
        for (int j = 1; j < seq_len; j++) {
            if (scores->data[i * seq_len + j] > max_val) {
                max_val = scores->data[i * seq_len + j];
            }
        }
        
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            scores->data[i * seq_len + j] = expf(scores->data[i * seq_len + j] - max_val);
            sum += scores->data[i * seq_len + j];
        }
        for (int j = 0; j < seq_len; j++) {
            scores->data[i * seq_len + j] /= sum;
        }
    }
    
    // Aplicar atención a V
    int scores_shape[2] = {seq_len, seq_len};
    Matrix *scores_mat = matrix_create_from_data(scores->data, seq_len, seq_len);
    Matrix *output = matrix_multiply(scores_mat, V_proj);
    
    // Convertir a tensor
    Tensor *result = tensor_create_from_data(output->data, q->shape, q->ndim);
    
    // Limpiar memoria
    matrix_free(Q); matrix_free(Q_proj);
    matrix_free(K); matrix_free(K_proj);
    matrix_free(V_mat); matrix_free(V_proj);
    tensor_free(Q_tensor); tensor_free(K_tensor);
    tensor_free(scores);
    matrix_free(scores_mat); matrix_free(output);
    
    return result;
}

/* ============================================================
 * FORWARD PASS CON TENSOR LOGIC
 * ============================================================ */

/**
 * Forward pass con lógica difusa
 * 
 * Combina:
 * 1. Transformación lineal
 * 2. Atención lógica
 * 3. Operaciones lógicas difusas
 * 4. Normalización
 */
Tensor* fuzzy_transformer_forward(FuzzyTransformerLayer *layer, Tensor *x) {
    int seq_len = x->shape[0];
    int d_model = x->shape[1];
    
    // 1. Atención lógica
    Tensor *attn_output = logical_attention(layer, x, x, x);
    
    // 2. Residual connection
    Tensor *residual1 = tensor_add(x, attn_output);
    
    // 3. Aplicar reglas lógicas
    Tensor *logic_output = apply_rules(layer, residual1);
    
    // 4. Residual connection
    Tensor *residual2 = tensor_add(residual1, logic_output);
    
    // 5. Layer normalization (simplificado)
    float mean = tensor_mean(residual2);
    float var = 0.0f;
    for (int i = 0; i < residual2->size; i++) {
        float diff = residual2->data[i] - mean;
        var += diff * diff;
    }
    var /= residual2->size;
    
    float std_inv = 1.0f / sqrtf(var + 1e-5f);
    Tensor *normalized = tensor_copy(residual2);
    for (int i = 0; i < normalized->size; i++) {
        normalized->data[i] = (normalized->data[i] - mean) * std_inv;
    }
    
    // Limpiar memoria
    tensor_free(attn_output);
    tensor_free(residual1);
    tensor_free(logic_output);
    tensor_free(residual2);
    
    return normalized;
}

/**
 * Forward pass con Markov Logic Network
 * 
 * Realiza inferencia en la MLN y retorna las predicciones
 */
Tensor* mln_forward(MarkovLogicNetwork *mln, Tensor *evidence) {
    // Realizar inferencia
    Tensor *truth_values = logical_inference(mln, evidence);
    
    // Convertir valores de verdad a logits
    Tensor *logits = tensor_create(evidence->shape, evidence->ndim);
    
    for (int i = 0; i < evidence->size; i++) {
        float truth = truth_values->data[i];
        // Mapear [0, 1] a [-∞, +∞] (logits)
        logits->data[i] = logf(truth / (1.0f - truth + 1e-7f));
    }
    
    tensor_free(truth_values);
    return logits;
}
