/**
 * DEMO_TENSOR_LOGIC.C - Demostración del Tensor Logic
 * 
 * Muestra cómo el Tensor Logic de Pedro Domingos
 * unifica lógica y aprendizaje en una sola estructura.
 */

#include "tensor_logic.h"

void print_separator() {
    printf("\n\033[1;36m");
    for (int i = 0; i < 70; i++) printf("═");
    printf("\033[0m\n");
}

void print_section(const char *title) {
    printf("\n\033[1;35m");
    for (int i = 0; i < 70; i++) printf("─");
    printf("\n  %s\n", title);
    for (int i = 0; i < 70; i++) printf("─");
    printf("\033[0m\n\n");
}

void demo_fuzzy_logic() {
    print_section("1. OPERACIONES LÓGICAS DIFUSAS");
    
    printf("  En lógica difusa, los valores de verdad están en [0, 1]\n");
    printf("  donde 0 = Falso, 1 = Verdadero, 0.5 = Incertidumbre\n\n");
    
    printf("  ┌─────────────────────────────────────────────────────────┐\n");
    printf("  │ Ejemplo: ¿Es Juan alto?                                │\n");
    printf("  │                                                         │\n");
    printf("  │ Literal: alto(Juan)                                     │\n");
    printf("  │ Valor de verdad: 0.7 (70% seguro de que es alto)       │\n");
    printf("  └─────────────────────────────────────────────────────────┘\n\n");
    
    // Demostrar t-normas
    printf("  T-Normas para AND lógico:\n");
    printf("  ─────────────────────────\n\n");
    
    float a = 0.7f, b = 0.8f;
    
    printf("  a = %.1f, b = %.1f\n\n", a, b);
    
    printf("  1. Producto:     a ⊗ b = a × b = %.2f\n", a * b);
    printf("  2. Łukasiewicz:  a ⊗ b = max(0, a+b-1) = %.2f\n", fmaxf(0.0f, a + b - 1.0f));
    printf("  3. Gödel:        a ⊗ b = min(a, b) = %.2f\n\n", fminf(a, b));
    
    // Demostrar OR
    printf("  T-Conormas (OR):\n");
    printf("  ─────────────────\n\n");
    
    printf("  1. Suma prob.:   a ⊕ b = a+b-a×b = %.2f\n", a + b - a * b);
    printf("  2. Łukasiewicz:  a ⊕ b = min(1, a+b) = %.2f\n", fminf(1.0f, a + b));
    printf("  3. Gödel:        a ⊕ b = max(a, b) = %.2f\n\n", fmaxf(a, b));
    
    // Demostrar NOT
    printf("  NOT (complemento):\n");
    printf("  ──────────────────\n\n");
    printf("  ¬a = 1 - a = %.2f\n\n", 1.0f - a);
    
    // Demostrar IMPLIES
    printf("  IMPLIES (implicación):\n");
    printf("  ──────────────────────\n\n");
    printf("  a → b = min(1, 1-a+b) = %.2f\n\n", fminf(1.0f, 1.0f - a + b));
}

void demo_literal_clause() {
    print_section("2. LITERALES Y CLÁUSULAS");
    
    // Crear literales
    printf("  Creando literales lógicos...\n\n");
    
    Literal *padre = literal_create("padre", 2, 1.5f);
    Literal *hijo = literal_create("hijo", 2, 1.2f);
    Literal *hermano = literal_create("hermano", 2, 0.8f);
    
    printf("  Literales creados:\n");
    printf("  ┌──────────┬─────────┬────────┬─────────────────────────┐\n");
    printf("  │ Nombre   │ Aritud  │ Peso   │ Embedding               │\n");
    printf("  ├──────────┼─────────┼────────┼─────────────────────────┤\n");
    printf("  │ %-8s │    %d    │  %.1f   │ [%.2f, %.2f]          │\n", 
           padre->name, padre->arity, padre->weight,
           padre->embedding->data[0], padre->embedding->data[1]);
    printf("  │ %-8s │    %d    │  %.1f   │ [%.2f, %.2f]          │\n",
           hijo->name, hijo->arity, hijo->weight,
           hijo->embedding->data[0], hijo->embedding->data[1]);
    printf("  │ %-8s │    %d    │  %.1f   │ [%.2f, %.2f]          │\n",
           hermano->name, hermano->arity, hermano->weight,
           hermano->embedding->data[0], hermano->embedding->data[1]);
    printf("  └──────────┴─────────┴────────┴─────────────────────────┘\n\n");
    
    // Crear cláusula
    printf("  Creando cláusula: padre(X,Y) ∧ hijo(Y,X) → hermano(X,Z)\n\n");
    
    Clause *clause = clause_create(3);
    clause->literals[0] = padre;
    clause->literals[1] = hijo;
    clause->literals[2] = hermano;
    clause->weight = 2.0f;
    
    printf("  Cláusula:\n");
    printf("    Peso: %.1f\n", clause->weight);
    printf("    Satisfacción: %.2f\n\n", clause->satisfaction);
}

void demo_mln() {
    print_section("3. MARKOV LOGIC NETWORK");
    
    printf("  Una Red de Markov Lógica combina:\n");
    printf("  - Reglas lógicas (cláusulas)\n");
    printf("  - Pesos aprendidos (incertidumbre)\n");
    printf("  - Inferencia probabilística\n\n");
    
    // Crear MLN
    printf("  Creando MLN con 3 cláusulas y 4 variables...\n\n");
    
    MarkovLogicNetwork *mln = markov_logic_create(3, 4, 2, TNORM_PRODUCT);
    
    printf("  Cláusulas:\n");
    printf("  ┌──────┬──────────────────────────────────────┬────────┐\n");
    printf("  │ ID   │ Literales                             │ Peso   │\n");
    printf("  ├──────┼──────────────────────────────────────┼────────┤\n");
    
    for (int i = 0; i < mln->n_clauses; i++) {
        Clause *c = mln->clauses[i];
        printf("  │  %d   │ ", i + 1);
        for (int j = 0; j < c->n_literals; j++) {
            printf("L%d", j + 1);
            if (j < c->n_literals - 1) printf(" ∧ ");
        }
        printf("%*s│  %.2f   │\n", 34 - c->n_literals * 4, "", c->weight);
    }
    printf("  └──────┴──────────────────────────────────────┴────────┘\n\n");
    
    // Valores iniciales
    printf("  Valores de verdad iniciales (incertidumbre):\n");
    printf("  ");
    for (int i = 0; i < mln->n_variables; i++) {
        printf("X%d=%.2f ", i + 1, mln->truth_values->data[i]);
    }
    printf("\n\n");
    
    // Inferencia
    printf("  Realizando inferencia...\n\n");
    
    Tensor *evidence = tensor_create((int[]){4}, 1);
    evidence->data[0] = 0.9f;  // X1 es casi seguro
    evidence->data[1] = 0.3f;  // X2 es poco probable
    evidence->data[2] = 0.7f;  // X2 es bastante probable
    evidence->data[3] = 0.5f;  // X3 es incierto
    
    printf("  Evidencia observada:\n");
    printf("  ");
    for (int i = 0; i < 4; i++) {
        printf("X%d=%.2f ", i + 1, evidence->data[i]);
    }
    printf("\n\n");
    
    Tensor *inferred = logical_inference(mln, evidence);
    
    printf("  Después de inferencia:\n");
    printf("  ");
    for (int i = 0; i < mln->n_variables; i++) {
        printf("X%d=%.2f ", i + 1, inferred->data[i]);
    }
    printf("\n\n");
    
    // Verificar consistencia
    float consistency = check_consistency(mln);
    printf("  Consistencia lógica: %.2f (%.0f%%)\n\n", 
           consistency, consistency * 100);
    
    tensor_free(evidence);
    tensor_free(inferred);
}

void demo_logical_attention() {
    print_section("4. ATENCIÓN LÓGICA");
    
    printf("  La atención lógica combina:\n");
    printf("  - Mecanismo de atención estándar\n");
    printf("  - Operaciones lógicas difusas\n");
    printf("  - Razonamiento sobre relaciones\n\n");
    
    // Crear capa
    FuzzyTransformerLayer *layer = fuzzy_transformer_create(64, 4, TNORM_PRODUCT);
    
    printf("  Capa creada:\n");
    printf("    - Dimensión: 64\n");
    printf("    - Predicados: 4\n");
    printf("    - T-norma: Producto\n\n");
    
    // Crear entrada
    int seq_len = 5;
    int d_model = 64;
    Tensor *input = tensor_create((int[]){seq_len, d_model}, 2);
    
    // Inicializar con valores aleatorios
    for (int i = 0; i < input->size; i++) {
        input->data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    printf("  Entrada: secuencia de %d tokens, dimensión %d\n\n", seq_len, d_model);
    
    // Forward pass
    printf("  Ejecutando forward pass con atención lógica...\n\n");
    
    Tensor *output = fuzzy_transformer_forward(layer, input);
    
    printf("  Salida generada: [%d, %d]\n\n", output->shape[0], output->shape[1]);
    
    // Mostrar valores de salida (primeros 5 elementos)
    printf("  Primeros valores de salida:\n  ");
    for (int i = 0; i < 5 && i < output->size; i++) {
        printf("%.3f ", output->data[i]);
    }
    printf("...\n");
    
    tensor_free(input);
    tensor_free(output);
}

void demo_tensor_vs_traditional() {
    print_section("5. TENSOR LOGIC vs TRANSFORMER TRADICIONAL");
    
    printf("  ┌─────────────────────┬─────────────────────┬─────────────────────┐\n");
    printf("  │ Característica      │ Transformer Trad.   │ Tensor Logic        │\n");
    printf("  ├─────────────────────┼─────────────────────┼─────────────────────┤\n");
    printf("  │ Atención            │ Score real          │ Score difuso        │\n");
    printf("  │ Operaciones         │ Aritméticas        │ Lógicas + Arit.     │\n");
    printf("  │ Incertidumbre       │ No explícita       │ Valores de verdad   │\n");
    printf("  │ Razonamiento        │ Implícito          │ Explícito (reglas)  │\n");
    printf("  │ Interpretabilidad   │ Baja               │ Alta                │\n");
    printf("  │ Datos faltantes     │ Problemático       │ Natural (difuso)    │\n");
    printf("  │ Conocimiento previo │ Difícil integrar   │ Fácil (cláusulas)   │\n");
    printf("  └─────────────────────┴─────────────────────┴─────────────────────┘\n\n");
    
    printf("  Ejemplo de razonamiento:\n\n");
    printf("  Transformer tradicional:\n");
    printf("    \"El gato come pescado\" → [0.2, 0.8, 0.1, ...]\n");
    printf("    (Solo patrones estadísticos)\n\n");
    printf("  Tensor Logic:\n");
    printf("    \"El gato come pescado\"\n");
    printf("    ∧ gato(X) → animal(X)    [regla conocida]\n");
    printf("    ∧ come(X,Y) → food(Y)    [regla conocida]\n");
    printf("    → animal(gato) = 0.95    [inferido]\n");
    printf("    → food(pescado) = 0.87   [inferido]\n");
    printf("    (Razonamiento + estadística)\n\n");
}

void demo_master_algorithm() {
    print_section("6. HACIA EL ALGORITMO MAESTRO");
    
    printf("  El Tensor Logic es un paso hacia el \"Algoritmo Maestro\"\n");
    printf("  de Pedro Domingos: un solo algoritmo que aprende todo.\n\n");
    
    printf("  Componentes del Algoritmo Maestro:\n\n");
    printf("  1. REPRESENTACIÓN: Redes de Markov Lógicas\n");
    printf("     └─ Combina lógica (simbólico) y probabilidad (numérico)\n\n");
    printf("  2. BÚSQUEDA: Gradient Descent + MCMC\n");
    printf("     └─ Aprende pesos de cláusulas y valores de verdad\n\n");
    printf("  3. EVALUACIÓN: Verosimilitud + Consistencia Lógica\n");
    printf("     └─ Optimiza ambos objetivos simultáneamente\n\n");
    
    printf("  ┌─────────────────────────────────────────────────────────────┐\n");
    printf("  │                                                             │\n");
    printf("  │   Datos ──→ [ Tensor Logic ] ──→ Conocimiento Aprendido   │\n");
    printf("  │                ↑         ↑                                  │\n");
    printf("  │            Lógica    Probabilidad                           │\n");
    printf("  │           (reglas)    (pesos)                               │\n");
    printf("  │                                                             │\n");
    printf("  └─────────────────────────────────────────────────────────────┘\n\n");
    
    printf("  Ventajas potenciales:\n");
    printf("  - Aprende con pocos datos (transfer learning natural)\n");
    printf("  - Razona sobre conceptos abstractos\n");
    printf("  - Maneja incertidumbre de forma principiada\n");
    printf("  - Interpretable: puede explicar sus decisiones\n\n");
}

int main() {
    tensor_seed(time(NULL));
    
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║                                                                  ║\n");
    printf("║          TENSOR LOGIC DE PEDRO DOMINGOS                          ║\n");
    printf("║          Unificación de Lógica y Aprendizaje                     ║\n");
    printf("║                                                                  ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    
    demo_fuzzy_logic();
    demo_literal_clause();
    demo_mln();
    demo_logical_attention();
    demo_tensor_vs_traditional();
    demo_master_algorithm();
    
    print_separator();
    printf("\n\033[1;32m  ✓ Demostración del Tensor Logic completada\033[0m\n\n");
    print_separator();
    
    return 0;
}
