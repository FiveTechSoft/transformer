/**
 * TENSOR_LOGIC.H - Tensor Logic de Pedro Domingos
 * 
 * Implementación del Tensor Logic que unifica:
 * - Lógica simbólica (reglas, relaciones)
 * - Aprendizaje numérico (pesos, gradientes)
 * - Inferencia probabilística (incertidumbre)
 * 
 * Referencia: Domingos, P. "The Master Algorithm" (2015)
 *             Domingos & Lowd "Markov Logic Networks" (2009)
 */

#ifndef TENSOR_LOGIC_H
#define TENSOR_LOGIC_H

#include "tensor.h"

/* ============================================================
 * ESTRUCTURAS DE TENSOR LOGIC
 * ============================================================ */

/**
 * Literal - Símbolo lógico con peso de confianza
 * Representa un átomo o predicado en la lógica
 */
typedef struct {
    char *name;             // Nombre del literal (ej: "padre", "hijo")
    int arity;              // Número de argumentos
    float weight;           // Peso de confianza [-∞, +∞]
    float truth_value;      // Valor de verdad [0, 1]
    Tensor *embedding;      // Representación vectorial del literal
} Literal;

/**
 * Clause - Cláusula lógica (regla)
 * Representa una regla del tipo: A ∧ B → C
 */
typedef struct {
    Literal **literals;     // Literales en la cláusula
    int n_literals;         // Número de literales
    float weight;           // Peso de la cláusula
    float satisfaction;     // Cuánto se satisface la cláusula [0, 1]
    
    // Tensor que codifica la relación lógica
    Tensor *relation_tensor;
} Clause;

/**
 * T-Norm - Norma tensorial para combinar evidencia
 * 
 * Implementa diferentes t-normas para combinar
 * valores de verdad de proposiciones:
 * - Producto: a ⊗ b = a * b
 * - Łukasiewicz: a ⊗ b = max(0, a + b - 1)
 * - Gödel: a ⊗ b = min(a, b)
 */
typedef enum {
    TNORM_PRODUCT,
    TNORM_LUKASIEWICZ,
    TNORM_GODEL
} TNormType;

/**
 * Markov Logic Network (simplificada)
 * Red de Markov Lógica que combina lógica y probabilidad
 */
typedef struct {
    Clause **clauses;       // Cláusulas lógicas
    int n_clauses;          // Número de cláusulas
    
    int n_variables;        // Número de variables lógicas
    int n_predicates;       // Número de predicados
    
    // Matriz de adjacencia lógica
    Tensor *adjacency;
    
    // Pesos aprendidos para cada cláusula
    Tensor *clause_weights;
    
    // Valores de verdad de las variables
    Tensor *truth_values;
    
    TNormType tnorm;        // Tipo de t-norma
} MarkovLogicNetwork;

/**
 * Tensor Logic Layer - Capa que aplica operaciones lógicas via tensores
 */
typedef struct {
    int input_dim;          // Dimensión de entrada
    int logic_dim;          // Dimensión del espacio lógico
    int n_predicates;       // Número de predicados lógicos
    
    // Pesos para proyección al espacio lógico
    Matrix *W_logic;
    
    // Pesos para operaciones lógicas (AND, OR, NOT, IMPLIES)
    Tensor *and_weights;
    Tensor *or_weights;
    Tensor *not_weights;
    Tensor *implies_weights;
    
    // Embeddings de predicados
    Matrix *predicate_embeddings;
    
    // T-norma a usar
    TNormType tnorm;
} TensorLogicLayer;

/**
 * Fuzzy Logic Transformer Layer
 * Transformer que opera con lógica difusa
 */
typedef struct {
    TensorLogicLayer *logic_layer;
    
    // Para atención lógica
    Matrix *W_q_logic;
    Matrix *W_k_logic;
    Matrix *W_v_logic;
    
    // Para razonamiento lógico
    Tensor *knowledge_base;  // Base de conocimiento
    
    // Operador de unión (para OR difuso)
    float (*union_op)(float a, float b);
    
    // Operador de intersección (para AND difuso)
    float (*intersection_op)(float a, float b, int tnorm);
} FuzzyTransformerLayer;

/* ============================================================
 * FUNCIONES DE TENSOR LOGIC
 * ============================================================ */

// Crear literal
Literal* literal_create(const char *name, int arity, float weight);

// Crear cláusula
Clause* clause_create(int n_literals);

// Crear capa de Tensor Logic
TensorLogicLayer* tensor_logic_layer_create(int input_dim, int logic_dim, 
                                             int n_predicates, TNormType tnorm);

// Crear Red de Markov Lógica
MarkovLogicNetwork* markov_logic_create(int n_clauses, int n_variables, 
                                        int n_predicates, TNormType tnorm);

// Crear capa Fuzzy Transformer
FuzzyTransformerLayer* fuzzy_transformer_create(int d_model, int n_predicates,
                                                 TNormType tnorm);

/* ============================================================
 * OPERACIONES LÓGICAS VIA TENSORES
 * ============================================================ */

// AND lógico difuso (intersección)
float fuzzy_and(float a, float b, TNormType tnorm);

// OR lógico difuso (unión)
float fuzzy_or(float a, float b);

// NOT lógico difuso (complemento)
float fuzzy_not(float a);

// IMPLIES lógico difuso
float fuzzy_implies(float a, float b);

// Aplicar t-norma a tensor
Tensor* apply_tnorm(Tensor *a, Tensor *b, TNormType tnorm);

/* ============================================================
 * RAZONAMIENTO LÓGICO
 * ============================================================ */

// Inferir nueva información a partir de reglas
Tensor* logical_inference(MarkovLogicNetwork *mln, Tensor *evidence);

// Aplicar reglas de inferencia
Tensor* apply_rules(FuzzyTransformerLayer *layer, Tensor *input);

// Verificar consistencia lógica
float check_consistency(MarkovLogicNetwork *mln);

/* ============================================================
 * ATENCIÓN LÓGICA
 * ============================================================ */

// Multi-Head Attention con operaciones lógicas
Tensor* logical_attention(FuzzyTransformerLayer *layer, 
                          Tensor *q, Tensor *k, Tensor *v);

// Calcular score de atención difuso
Tensor* fuzzy_attention_score(Tensor *q, Tensor *k, TNormType tnorm);

/* ============================================================
 * FORWARD PASS CON TENSOR LOGIC
 * ============================================================ */

// Forward pass con lógica difusa
Tensor* fuzzy_transformer_forward(FuzzyTransformerLayer *layer, Tensor *x);

// Forward pass con Markov Logic Network
Tensor* mln_forward(MarkovLogicNetwork *mln, Tensor *evidence);

#endif /* TENSOR_LOGIC_H */
