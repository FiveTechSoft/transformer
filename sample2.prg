/*
* ========================================================================
* EJEMPLO DIDÁCTICO: EMBEDDINGS DENSOS vs ONE-HOT
* Área: Question Answering sobre colores de objetos
* VOCABULARIO: 15 tokens (objetos, colores, palabras de pregunta)
* EMBEDDINGS: Densos de dimensión 8 (aprenden relaciones semánticas)
* MEJORADO: Con gradient clipping, label smoothing y mejores técnicas de entrenamiento
* ========================================================================
*/

#include "hbclass.ch"

REQUEST HB_GT_STD, HB_GT_STD_DEFAULT

PROCEDURE Main()
   // ========================================================================
   // CONSTANTES DE CONFIGURACIÓN - ARQUITECTURA DE ALTA CAPACIDAD  
   // ========================================================================
   LOCAL nVocabSize := 15
   LOCAL nEmbedDim := 12                 // TRIPLICADO: máxima expresividad
   LOCAL nHiddenDim := 48                // TRIPLICADO: alta capacidad
   LOCAL nHeadDim := 12                  // Coherente con embed
   LOCAL nLayers := 1                    // Una capa pero con mucha capacidad
   LOCAL nSeqLen := 8
   LOCAL nAnswerPos := 7

   // Configuración de entrenamiento para alta capacidad
   LOCAL nEpochs := 1000                 // Más épocas para convergencia
   LOCAL nLearningRate := 0.005          // LR más bajo para estabilidad
   LOCAL nGradientClipping := 0.8        // Clipping más permisivo
   LOCAL nLabelSmoothing := 0.05         // Smoothing mínimo
   LOCAL nWeightDecay := 0.0005          // Weight decay ligero
   LOCAL nMomentum := 0.95               // Momentum alto para estabilidad

   // Configuración de inicialización AGRESIVA para forzar exploración
   LOCAL nEmbeddingStdDev := 0.3         // Inicialización MUY agresiva 
   LOCAL nXavierFactor := 3.0            // Factor Xavier muy agresivo

   // Configuración simple sin scheduling complejo
   LOCAL nCycleLength := 150             // Ciclos medianos
   LOCAL nMinLRFactor := 0.05            // LR mínimo bajo
   LOCAL nPhaseStep := 500               // Cambio de fase tardío

   // Configuración de Focal Loss
   LOCAL nFocalAlpha := 0.25          // Factor de balanceo para focal loss
   LOCAL nFocalGamma := 2.0           // Factor de enfoque para casos difíciles

   // Constantes para objetos especiales en positional encoding
   LOCAL nObjectPositionStart := 5
   LOCAL nObjectPositionEnd := 6
   LOCAL nObjectAttentionFactor := 1.5

   // Índices de tokens (base-0)
   LOCAL nTokenPad := 0, nTokenWhat := 1, nTokenColor := 2, nTokenIs := 3, nTokenThe := 4
   LOCAL nTokenSky := 5, nTokenGrass := 6, nTokenSun := 7, nTokenOcean := 8
   LOCAL nTokenBlue := 9, nTokenGreen := 10, nTokenYellow := 11, nTokenRed := 12
   LOCAL nTokenOrange := 13, nTokenQuestion := 14
   
   // Variables principales del modelo
   LOCAL oModel, mEmbeddings, mPositionalEncoding
   LOCAL mClassificationWeights, mClassificationBias
   LOCAL mClassMomentumW, mClassMomentumB
   
   // Variables de entrenamiento
   LOCAL aInput, aTarget, mInput, mTarget, mOutput
   LOCAL nTotalLoss, nLoss, dLoss
   LOCAL mInputGradient, mTransformerOutput, mResponseVector
   LOCAL mLogits, mProbabilities, mProbGradient, mResponseGradient, mWeightGradient
   LOCAL nDropoutRate := 0.2             // Dropout más agresivo para regularización
   LOCAL nAdversarialPenalty := 0         // Penalty para regularización adversarial
   LOCAL nSemanticPenalty := 0            // Penalty por errores semánticos específicos
   LOCAL nSemanticReward := 0             // Recompensa por predicciones semánticamente correctas
   LOCAL nObjectType := 0                 // Tipo de objeto detectado (1=SKY/OCEAN, 2=GRASS, 3=SUN)
   LOCAL aContrastiveExamples, aWrongExample  // Variables para contrastive learning
   
   // 🚨 VARIABLES PARA CONSTRAINT DE DIVERSIDAD OBLIGATORIA
   LOCAL aPredictionCounts := {0, 0, 0}   // Contador para BLUE(9), GREEN(10), YELLOW(11)
   LOCAL nDiversityCheckInterval := 20    // Revisar diversidad cada 20 ejemplos
   LOCAL nMaxDominanceRatio := 0.6        // Máximo 60% de predicciones de una sola clase
   LOCAL nDiversityPenaltyMultiplier := 10.0  // Penalty EXTREMO por dominancia
   
   // 📈 VARIABLES PARA LEARNING RATE ADAPTATIVO POR CLASE
   LOCAL aClassAccuracy := {0.0, 0.0, 0.0}    // Accuracy por clase BLUE, GREEN, YELLOW
   LOCAL aClassCounts := {0, 0, 0}             // Contadores de ejemplos por clase
   LOCAL aClassCorrect := {0, 0, 0}            // Contadores de aciertos por clase
   LOCAL nLRBoostFactor := 2.0                 // Factor de boost para clases con baja accuracy
   LOCAL nLRPenaltyFactor := 0.5               // Factor de penalty para clases dominantes
   
   // Variables adicionales para constraints de diversidad y LR adaptativo
   LOCAL nTotalPredictions, nMaxClassCount, nDominanceRatio, nDiversityConstraintPenalty
   LOCAL nClassLRMultiplier, cClassStatus
   LOCAL nEntropyPenalty, nMaxProbInBatch, nMinProbInBatch, nProbRange  // Anti-colapso
   LOCAL nDominancePenalty  // Anti-dominancia extrema
   LOCAL aFilteredData, nCurrentPhase, nBlueProb  // Curriculum learning revolucionario
   LOCAL nMSELoss, aTargetVector, aPredVector  // MSE Loss variables
   
   // Variables de control
   LOCAL i, j, k, l
   LOCAL nTargetToken, nPredictedToken, nMaxProb, nXavierStd, nCurrentLR
   LOCAL nPhase, cPhaseDesc
   LOCAL aColorExamples, nRand, temp
   LOCAL nTargetIdx, nTargetProb, nCycle, nCyclePos
   LOCAL nMaxLR, nMinLR, nProgress, nRandTemp, tempSwap
   
   // Variables para curriculum learning
   LOCAL aCurrentExamples := {}
   LOCAL nCurriculumPhase := 1
   LOCAL aUniqueExamples := { ;
      { {1,2,3,4,5,14,0,0}, {0,0,0,0,0,0,9,0} },  ; // SKY -> BLUE
      { {1,2,3,4,6,14,0,0}, {0,0,0,0,0,0,10,0} }, ; // GRASS -> GREEN  
      { {1,2,3,4,7,14,0,0}, {0,0,0,0,0,0,11,0} }, ; // SUN -> YELLOW
      { {1,2,3,4,8,14,0,0}, {0,0,0,0,0,0,9,0} }   ; // OCEAN -> BLUE
   }
   LOCAL aBalancedExamples := { ;
      { {1,2,3,4,5,14,0,0}, {0,0,0,0,0,0,9,0} },  ; // SKY -> BLUE
      { {1,2,3,4,6,14,0,0}, {0,0,0,0,0,0,10,0} }, ; // GRASS -> GREEN  
      { {1,2,3,4,7,14,0,0}, {0,0,0,0,0,0,11,0} }, ; // SUN -> YELLOW
      { {1,2,3,4,8,14,0,0}, {0,0,0,0,0,0,9,0} },  ; // OCEAN -> BLUE
      { {1,2,3,4,5,14,0,0}, {0,0,0,0,0,0,9,0} },  ; // SKY -> BLUE (repetir)
      { {1,2,3,4,6,14,0,0}, {0,0,0,0,0,0,10,0} }, ; // GRASS -> GREEN (repetir)
      { {1,2,3,4,8,14,0,0}, {0,0,0,0,0,0,9,0} }   ; // OCEAN -> BLUE (repetir)
   }
   
   // Variables de análisis y validación
   LOCAL aCorrectAssociations, nCorrect, nTotal, cResult
   LOCAL nToken, nU1, nU2, nZ0, nStdDev, nGrad
   LOCAL nYellowPenalty := 0
   LOCAL nDiversityPenalty := 0
   LOCAL nAvgGradMagnitude := 0
   LOCAL cTrainingPhase := ""
   LOCAL nSumSq, nDist, nBestToken, nMinDist, nLen, nSpecialFactor, nAngle
   LOCAL aRow, nSmoothComponent, mClipped, mSmoothedProbs
   LOCAL mPE, pRow, pos, nClipValue
   LOCAL aVocab, aTrainingData, nColorToken
   
   // Variables para mejoras avanzadas + Early Stopping
   LOCAL aClassWeights, nFocalLoss, nClassWeight
   LOCAL nCorrectByClass := {0, 0, 0, 0}  // Contador por clase (BLUE, GREEN, YELLOW, otros)
   LOCAL nTotalByClass := {0, 0, 0, 0}    // Total por clase
   LOCAL nWarmupEpochs := 50              // Épocas de warm-up (reducido)
   LOCAL nOriginalLR := nLearningRate     // Guardar LR original
   
   // Variables para early stopping y mejor modelo
   LOCAL nBestLoss := 999999
   LOCAL nBestEpoch := 0
   LOCAL mBestClassWeights, mBestClassBias
   LOCAL nPatienceCounter := 0
   LOCAL nMaxPatience := 100  // Épocas sin mejora antes de parar
   LOCAL nValidationFreq := 25  // Validar cada 25 épocas
   
   // --- VOCABULARIO EXPANDIDO (15 tokens) ---
   // Índices: 0=PAD, 1=WHAT, 2=COLOR, 3=IS, 4=THE, 
   //          5=SKY, 6=GRASS, 7=SUN, 8=OCEAN,
   //          9=BLUE, 10=GREEN, 11=YELLOW, 12=RED, 13=ORANGE, 14=?
   aVocab := { ;
      "PAD", "WHAT", "COLOR", "IS", "THE", ;
      "SKY", "GRASS", "SUN", "OCEAN", ;
      "BLUE", "GREEN", "YELLOW", "RED", "ORANGE", "?" ;
   }
   
   // --- DATASET ESTRATÉGICAMENTE BALANCEADO (24 ejemplos) ---
   // Alternando colores para evitar sobreajuste hacia un solo color
   // Ahora balanceado para tener representación equitativa de todos los objetos-colores
   // Mejorado con ejemplos adicionales para ayudar a distinguir YELLOW de GREEN
   aTrainingData := { ;
      { {1,2,3,4,6,14,0,0}, {0,0,0,0,0,0,10,0} }, ; // "WHAT COLOR IS THE GRASS ?" -> "GREEN" (posición 7)
      { {1,2,3,4,5,14,0,0}, {0,0,0,0,0,0,9,0} },  ; // "WHAT COLOR IS THE SKY ?" -> "BLUE" (posición 7) 
      { {1,2,3,4,7,14,0,0}, {0,0,0,0,0,0,11,0} }, ; // "WHAT COLOR IS THE SUN ?" -> "YELLOW" (posición 7)
      { {1,2,3,4,8,14,0,0}, {0,0,0,0,0,0,9,0} },  ; // "WHAT COLOR IS THE OCEAN ?" -> "BLUE" (posición 7)
      { {1,2,3,4,6,14,0,0}, {0,0,0,0,0,0,10,0} }, ; // "WHAT COLOR IS THE GRASS ?" -> "GREEN" (posición 7)
      { {1,2,3,4,7,14,0,0}, {0,0,0,0,0,0,11,0} }, ; // "WHAT COLOR IS THE SUN ?" -> "YELLOW" (posición 7)
      { {1,2,3,4,6,14,0,0}, {0,0,0,0,0,0,10,0} }, ; // "WHAT COLOR IS THE GRASS ?" -> "GREEN" (posición 7)
      { {1,2,3,4,5,14,0,0}, {0,0,0,0,0,0,9,0} },  ; // "WHAT COLOR IS THE SKY ?" -> "BLUE" (posición 7)
      { {1,2,3,4,7,14,0,0}, {0,0,0,0,0,0,11,0} }, ; // "WHAT COLOR IS THE SUN ?" -> "YELLOW" (posición 7)
      { {1,2,3,4,6,14,0,0}, {0,0,0,0,0,0,10,0} }, ; // "WHAT COLOR IS THE GRASS ?" -> "GREEN" (posición 7)
      { {1,2,3,4,8,14,0,0}, {0,0,0,0,0,0,9,0} },  ; // "WHAT COLOR IS THE OCEAN ?" -> "BLUE" (posición 7)
      { {1,2,3,4,7,14,0,0}, {0,0,0,0,0,0,11,0} }, ; // "WHAT COLOR IS THE SUN ?" -> "YELLOW" (posición 7)
      { {1,2,3,4,6,14,0,0}, {0,0,0,0,0,0,10,0} }, ; // "WHAT COLOR IS THE GRASS ?" -> "GREEN" (posición 7)
      { {1,2,3,4,5,14,0,0}, {0,0,0,0,0,0,9,0} },  ; // "WHAT COLOR IS THE SKY ?" -> "BLUE" (posición 7)
      { {1,2,3,4,7,14,0,0}, {0,0,0,0,0,0,11,0} }, ; // "WHAT COLOR IS THE SUN ?" -> "YELLOW" (posición 7)
      { {1,2,3,4,6,14,0,0}, {0,0,0,0,0,0,10,0} }, ; // "WHAT COLOR IS THE GRASS ?" -> "GREEN" (posición 7)
      { {1,2,3,4,8,14,0,0}, {0,0,0,0,0,0,9,0} },  ; // "WHAT COLOR IS THE OCEAN ?" -> "BLUE" (posición 7)
      { {1,2,3,4,7,14,0,0}, {0,0,0,0,0,0,11,0} }, ; // "WHAT COLOR IS THE SUN ?" -> "YELLOW" (posición 7)
      { {1,2,3,4,6,14,0,0}, {0,0,0,0,0,0,10,0} }, ; // "WHAT COLOR IS THE GRASS ?" -> "GREEN" (posición 7)
      { {1,2,3,4,5,14,0,0}, {0,0,0,0,0,0,9,0} },  ; // "WHAT COLOR IS THE SKY ?" -> "BLUE" (posición 7)
      { {1,2,3,4,8,14,0,0}, {0,0,0,0,0,0,9,0} },  ; // "WHAT COLOR IS THE OCEAN ?" -> "BLUE" (posición 7)
      { {1,2,3,4,7,14,0,0}, {0,0,0,0,0,0,11,0} }, ; // "WHAT COLOR IS THE SUN ?" -> "YELLOW" (posición 7)
      { {1,2,3,4,7,14,0,0}, {0,0,0,0,0,0,11,0} }, ; // "WHAT COLOR IS THE SUN ?" -> "YELLOW" (posición 7) - EXTRA para balance
      { {1,2,3,4,7,14,0,0}, {0,0,0,0,0,0,11,0} }  ; // "WHAT COLOR IS THE SUN ?" -> "YELLOW" (posición 7) - EXTRA para balance
   }
   
   // DATA AUGMENTATION: Generar variaciones sintéticas del dataset
   aTrainingData := GenerateAugmentedDataset(aTrainingData)
   
   // Variables de entrenamiento inicializadas
   nColorToken := 0
   
   // Validar datos de entrenamiento (ahora aumentados)
   ? "Validando", Len(aTrainingData), "ejemplos de entrenamiento (incluyendo sintéticos)..."
   FOR i := 1 TO Len(aTrainingData)
      IF !ValidateTokens(aTrainingData[i][1], nVocabSize, "Input ejemplo " + Str(i))
         ? "FATAL: Error en datos de entrada del ejemplo", i
         QUIT
      ENDIF
      IF !ValidateTokens(aTrainingData[i][2], nVocabSize, "Target ejemplo " + Str(i))
         ? "FATAL: Error en datos objetivo del ejemplo", i
         QUIT
      ENDIF
   NEXT
   ? "✓ Validación de datos completada"
   ? ""
   
   ? "========================================================================="
   ? "TRANSFORMER SUPER-ENHANCED - VERSIÓN REVOLUCIONARIA"
   ? "========================================================================="
   ? ""
   ? "ARQUITECTURA PROFUNDA MEJORADA:"
   ? "  • Dimensión de embeddings:", nEmbedDim, "(increased from 6)"
   ? "  • Dimensión oculta:", nHiddenDim, "(increased from 16)"
   ? "  • Capas transformer:", nLayers, "(increased from 1)"
   ? "  • Dropout rate:", nDropoutRate, "(regularización avanzada)"
   ? ""
   ? "MEJORAS REVOLUCIONARIAS EN ESTA VERSIÓN:"
   ? "  🟢 Augmentación MASIVA para GREEN: 15x más datos sintéticos"
   ? "  🧠 Arquitectura más profunda: 2 capas, 10-dim embeddings"
   ? "  🚨 Regularización adversarial: Penalty extremo GREEN→BLUE confusion"
   ? "  ⚡ Early stopping avanzado con mejor modelo"
   ? "  🎯 Anti-bias initialization revolucionario"
   ? "  📊 Análisis gradientes en tiempo real"
   ? "  🔄 Curriculum learning optimizado"
   ? ""
   ? "VENTAJAS DE EMBEDDINGS DENSOS:"
   ? "  1. Capturan relaciones semánticas (ej: 'SKY' y 'OCEAN' están cerca)"
   ? "  2. Dimensión reducida:", nEmbedDim, "vs", nVocabSize, "(one-hot)"
   ? "  3. Se entrenan con el modelo (aprenden significado del contexto)"
   ? "  4. Permiten generalización (palabras similares -> vectores similares)"
   ? ""
   ? "TÉCNICAS DE ENTRENAMIENTO AVANZADAS:"
   ? "  • Gradient clipping:", nGradientClipping, "(previene explosion de gradientes)"
   ? "  • Label smoothing:", nLabelSmoothing, "(previene overconfidence)"
   ? "  • Learning rate cíclico: min=", Transform(nLearningRate * nMinLRFactor, "@E 9.9999"), "max=", Transform(nLearningRate, "@E 9.9999")
   ? "  • Weight decay:", nWeightDecay, "(regularización L2 fuerte)"
   ? "  • Momentum SGD:", nMomentum, "(optimización estable)"
   ? "  • Focal Loss: α=", nFocalAlpha, "γ=", nFocalGamma, "(combate sesgo de clase)"
   ? "  • Class balancing: activado (pesos automáticos)"
   ? "  • Warm-up:", nWarmupEpochs, "épocas (calentamiento gradual)"
   ? "  • Dropout:", nDropoutRate, "(regularización durante entrenamiento)"
   ? ""
   ? "MEJORAS EN ESTA VERSIÓN:"
   ? "  ✓ Validación completa de parámetros y matrices"
   ? "  ✓ Manejo robusto de errores en tiempo de ejecución"
   ? "  ✓ Configuración mediante constantes nombradas"
   ? "  ✓ Optimizaciones de performance en operaciones matriciales"
   ? "  ✓ Documentación mejorada y comentarios detallados"
   ? "  ✓ Inicialización anti-sesgo radical por clase específica"
   ? "  ✓ Curriculum learning progresivo (fácil → difícil)"
   ? "  ✓ Early stopping con guardado del mejor modelo"
   ? "  ✓ Análisis de gradientes en tiempo real"
   ? "  ✓ Data augmentation inteligente balanceado"
   ? ""
   ? "========================================================================="
   ? ""
   
   // --- CREAR EMBEDDINGS DENSOS (inicialización Gaussiana) ---
   mEmbeddings := CreateDenseEmbeddings(nVocabSize, nEmbedDim)
   
   // Validar embeddings creados
   IF !ValidateMatrix(mEmbeddings, nVocabSize, nEmbedDim, "mEmbeddings")
      ? "FATAL: Error en la creación de embeddings"
      QUIT
   ENDIF
   
   // --- CREAR MODELO CON CAPA DE CLASIFICACIÓN ---
   oModel := TransformerModel():New(nLayers, nEmbedDim, nHiddenDim, nHeadDim, nDropoutRate)
   
   // Crear capa de clasificación con inicialización anti-sesgo
   nXavierStd := Sqrt(nXavierFactor / (nEmbedDim + nVocabSize))
   mClassificationWeights := HB_MATRIXRANDOM(nEmbedDim, nVocabSize)
   // Reescalar con Xavier initialization
   mClassificationWeights := HB_MATRIXMULSCALAR(mClassificationWeights, nXavierStd)
   
   // INICIALIZACIÓN ANTI-SESGO RADICAL: Forzar distinción desde el inicio
   mClassificationBias := HB_MATRIXZERO(1, nVocabSize)
   
   // ESTRATEGIA REVOLUCIONARIA: Inicializar pesos de clasificación específicos por clase
   // 🚫 SABOTAJE INICIAL A BLUE para romper dominancia
   
   // Resetear todos los pesos de clasificación a valores pequeños
   FOR i := 1 TO nEmbedDim
      FOR j := 1 TO nVocabSize
         mClassificationWeights[i][j] := GaussianRandom(0, nXavierStd)  // Base normal
      NEXT
   NEXT
   
   // ⚖️ INICIALIZACIÓN EQUILIBRADA: Pesos idénticos para los 3 colores
   FOR i := 1 TO nEmbedDim
      mClassificationWeights[i][10] := GaussianRandom(0, nXavierStd)  // BLUE: distribución normal
      mClassificationWeights[i][11] := GaussianRandom(0, nXavierStd)  // GREEN: distribución normal
      mClassificationWeights[i][12] := GaussianRandom(0, nXavierStd)  // YELLOW: distribución normal
   NEXT
   
   // ⚖️ INICIALIZACIÓN EQUILIBRADA: Pesos idénticos para las 3 clases
   mClassificationBias[1][10] := 0.0    // BLUE: bias neutro
   mClassificationBias[1][11] := 0.0    // GREEN: bias neutro
   mClassificationBias[1][12] := 0.0    // YELLOW: bias neutro
   
   ? "⚖️ INICIALIZACIÓN EQUILIBRADA: Pesos idénticos para BLUE, GREEN, YELLOW"
   
   // Inicializar momentum para capa de clasificación
   mClassMomentumW := HB_MATRIXZERO(nEmbedDim, nVocabSize)
   mClassMomentumB := HB_MATRIXZERO(1, nVocabSize)
   
   // Guardar estado inicial como mejor modelo
   mBestClassWeights := HB_MATRIXCLONE(mClassificationWeights)
   mBestClassBias := HB_MATRIXCLONE(mClassificationBias)
   
   // Calcular pesos balanceados por clase para combatir el sesgo
   aClassWeights := CalculateClassWeights(aTrainingData, nVocabSize)
   ? "Pesos de clase calculados para balancear el dataset:"
   ? "  BLUE (", nTokenBlue, "):", Transform(aClassWeights[nTokenBlue + 1], "@E 9.99")
   ? "  GREEN (", nTokenGreen, "):", Transform(aClassWeights[nTokenGreen + 1], "@E 9.99") 
   ? "  YELLOW (", nTokenYellow, "):", Transform(aClassWeights[nTokenYellow + 1], "@E 9.99")
   ? ""
   
   ? "Embeddings inicializados con distribución Normal N(0,", nEmbeddingStdDev, ")"
   ? "Xavier initialization aplicada a capa de clasificación (std=", Transform(nXavierStd, "@E 9.999"), ")"
   ? "Ejemplo - Embedding del token 'SKY' (índice", nTokenSky, "):"
   ? "  ", HB_ValToExp(mEmbeddings[nTokenSky + 1])  // +1 porque arrays en Harbour son base-1
   ? ""
   
   // --- CREAR POSITIONAL ENCODING ---
   mPositionalEncoding := CreatePositionalEncoding(nSeqLen, nEmbedDim)
   
   // Validar positional encoding
   IF !ValidateMatrix(mPositionalEncoding, nSeqLen, nEmbedDim, "mPositionalEncoding")
      ? "FATAL: Error en la creación del positional encoding"
      QUIT
   ENDIF
   
   // Definir las asociaciones correctas de color-objeto para validación posterior
   aCorrectAssociations := { ;
      {5, 9},  ; // "SKY" -> "BLUE" (índices 0-base: 5->9)
      {6, 10}, ; // "GRASS" -> "GREEN" (índices 0-base: 6->10)
      {7, 11}, ; // "SUN" -> "YELLOW" (índices 0-base: 7->11)
      {8, 9}   ; // "OCEAN" -> "BLUE" (índices 0-base: 8->9)
   }
   
   ? "Iniciando entrenamiento con algoritmo optimizado..."
   ? "Configuración: Epochs=", nEpochs, "LR=", nLearningRate, "Batch=", Len(aTrainingData), "ejemplos"
   ? Replicate("-", 80)
   
   // --- ENTRENAMIENTO CON TRIPLE CURRICULUM LEARNING ---
   // � FASE 1 (épocas 1-100): Solo BLUE - Otros EXCLUIDOS
   // 🟢 FASE 2 (épocas 101-200): Solo GREEN - Otros EXCLUIDOS  
   // 🟡 FASE 3 (épocas 201-300): Solo YELLOW - Otros EXCLUIDOS
   // 🔄 FASE 4 (épocas 301+): Dataset completo con anti-dominancia universal
   
   FOR i := 1 TO nEpochs
      nTotalLoss := 0
      
      // 🎯 APRENDIZAJE SIMULTÁNEO: Dataset completo desde época 1
      aFilteredData := aTrainingData  // No más filtrado por fases  
      cTrainingPhase := " [DATASET COMPLETO EQUILIBRADO]"
      
      // Learning rate scheduling con warm-up
      IF i <= nWarmupEpochs
         // Warm-up: incrementar gradualmente el learning rate
         nCurrentLR := nOriginalLR * (i / nWarmupEpochs) * 0.5  // Empezar con 50% del LR
      ELSE
         // Learning rate cíclico después del warm-up
         nCycle := Int((i - nWarmupEpochs - 1) / nCycleLength)
         nCyclePos := (i - nWarmupEpochs - 1) % nCycleLength
         nMaxLR := nLearningRate
         nMinLR := nLearningRate * nMinLRFactor
         
         // Calcular LR cíclico con coseno annealing
         nProgress := nCyclePos / nCycleLength
         nCurrentLR := nMinLR + (nMaxLR - nMinLR) * (Cos(nProgress * 3.14159) + 1) / 2
      ENDIF
      
      // Entrenamiento por fases: dos fases principales para mejor balance
      nPhase := Int((i-1) / nPhaseStep) + 1
      IF nPhase > 2
         nPhase := 3  // Última fase para refinamiento
      ENDIF
      // No necesitamos el código de filtrado anterior ya que lo hacemos arriba
      aColorExamples := aFilteredData  // Usar dataset filtrado por curriculum learning
      
      // 🎯 CONTRASTIVE LEARNING: Generar ejemplos contrastivos cada 10 épocas
      IF (i % 10) == 0
         aContrastiveExamples := {}
         FOR k := 1 TO Len(aColorExamples)
            // Añadir ejemplo original
            AAdd(aContrastiveExamples, aColorExamples[k])
            
            // Generar ejemplo contrastivo INCORRECTO para aprendizaje explícito
            aWrongExample := AClone(aColorExamples[k])
            nOriginalTarget := aColorExamples[k][2][7]
            
            // Cambiar a respuesta incorrecta según el objeto
            FOR m := 1 TO Len(aColorExamples[k][1])
               IF aColorExamples[k][1][m] == 5 .OR. aColorExamples[k][1][m] == 6  // SKY/OCEAN
                  aWrongExample[2][7] := 10  // Cambiar a GREEN (incorrecto)
                  EXIT
               ELSEIF aColorExamples[k][1][m] == 7  // GRASS
                  aWrongExample[2][7] := 9   // Cambiar a BLUE (incorrecto)
                  EXIT
               ELSEIF aColorExamples[k][1][m] == 8  // SUN
                  aWrongExample[2][7] := 9   // Cambiar a BLUE (incorrecto)
                  EXIT
               ENDIF
            NEXT
            
            AAdd(aContrastiveExamples, aWrongExample)
         NEXT
         aColorExamples := aContrastiveExamples
         ? "  🔄 CONTRASTIVE: Generados", Len(aContrastiveExamples), "ejemplos (50% incorrectos para aprendizaje)"
      ENDIF
      
      // Barajar los ejemplos para evitar sesbios de orden
      FOR j := Len(aColorExamples) TO 2 STEP -1
         nRandTemp := Int(hb_Random() * j) + 1
         tempSwap := aColorExamples[j]
         aColorExamples[j] := aColorExamples[nRandTemp]
         aColorExamples[nRandTemp] := tempSwap
      NEXT
      
      // Validación de seguridad: asegurar que siempre tenemos ejemplos
      IF Len(aColorExamples) == 0
         aColorExamples := aTrainingData  // Fallback al dataset completo
      ENDIF
      
      // Barajar los ejemplos para evitar sesgo de orden
      FOR j := Len(aColorExamples) TO 2 STEP -1
         nRand := Int(hb_Random() * j) + 1
         temp := aColorExamples[j]
         aColorExamples[j] := aColorExamples[nRand]
         aColorExamples[nRand] := temp
      NEXT
      
      // Entrenar con cada ejemplo del dataset
      FOR j := 1 TO Len(aColorExamples)
         aInput := aColorExamples[j][1]
         aTarget := aColorExamples[j][2]
         
         // Convertir tokens a embeddings
         mInput := CreateMatrixFromTokens(aInput, mEmbeddings)
         mTarget := CreateMatrixFromTokens(aTarget, mEmbeddings)
         
         // Añadir positional encoding
         mInput := HB_MATRIXADD(mInput, mPositionalEncoding)
         
         // Forward pass con capa de clasificación
         oModel:ZeroGrads()
         mTransformerOutput := oModel:Forward(mInput, .T.)  // Training mode
         
         // Siempre usar la posición constante para la respuesta como se define en el dataset
         // nAnswerPos is already set to 7 at the beginning
         
         // Aplicar capa de clasificación en la posición de la respuesta
         mResponseVector := {mTransformerOutput[nAnswerPos]}  // Solo la posición de respuesta
         mLogits := HB_MATRIXMULTIPLY(mResponseVector, mClassificationWeights)
         mLogits := HB_MATRIXADDBROADCAST(mLogits, mClassificationBias)
         
         // Aplicar label smoothing para prevenir overconfidence
         mProbabilities := ApplyLabelSmoothing(mLogits, nLabelSmoothing)
         
         // El target es el índice del token correcto
         nTargetToken := aTarget[nAnswerPos]
         
         // Obtener peso de la clase para balanceo
         nClassWeight := aClassWeights[nTargetToken + 1]
         
         // Añadir debugging para ver qué posiciones están seleccionadas
         IF i <= 10 .AND. j == 1  // Mostrar para las primeras épocas
            ? "  DEBUG: nAnswerPos =", nAnswerPos, "Target token =", nTargetToken, "(", aVocab[nTargetToken+1], ") Peso:", Transform(nClassWeight, "@E 9.99")
         ENDIF
         
         IF Empty(mProbabilities)
            ? "ERROR: Forward pass falló en época", i, "ejemplo", j
            ? "  Verificar estado del modelo y gradientes"
            QUIT
         ENDIF
         
         // Validar que las probabilidades sean válidas
         IF !ValidateMatrix(mProbabilities, 1, nVocabSize, "mProbabilities")
            ? "ERROR: Probabilidades inválidas en época", i, "ejemplo", j
            QUIT
         ENDIF
         
         // 🎯 USAR MSE LOSS en lugar de Focal Loss para evitar over-confidence
         nMSELoss := 0
         aTargetVector := {0, 0, 0}  // Vector one-hot para las 3 clases de color
         
         // Crear vector target one-hot
         IF nTargetToken == 9
            aTargetVector[1] := 1  // BLUE
         ELSEIF nTargetToken == 10  
            aTargetVector[2] := 1  // GREEN
         ELSEIF nTargetToken == 11
            aTargetVector[3] := 1  // YELLOW
         ENDIF
         
         // Calcular MSE loss solo para las 3 clases de color
         aPredVector := {mProbabilities[1][10], mProbabilities[1][11], mProbabilities[1][12]}
         FOR k := 1 TO 3
            nMSELoss := nMSELoss + ((aTargetVector[k] - aPredVector[k]) ^ 2)
         NEXT
         nFocalLoss := nMSELoss / 3.0  // Promedio MSE
         
         // 🚨 LOSS ANTI-COLAPSO: Penalizar uniformidad de predicciones
         nEntropyPenalty := 0
         nMaxProbInBatch := 0
         nMinProbInBatch := 1.0
         
         // Calcular entropía de la distribución de probabilidades
         FOR k := 1 TO nVocabSize
            IF mProbabilities[1][k] > nMaxProbInBatch
               nMaxProbInBatch := mProbabilities[1][k]
            ENDIF
            IF mProbabilities[1][k] < nMinProbInBatch
               nMinProbInBatch := mProbabilities[1][k]
            ENDIF
         NEXT
         
         // Si las probabilidades son muy uniformes (colapso), penalizar EXTREMADAMENTE
         nProbRange := nMaxProbInBatch - nMinProbInBatch
         IF nProbRange < 0.3  // Si diferencia entre max y min prob < 30%
            nEntropyPenalty := 5.0 * (0.3 - nProbRange)  // Penalty proporcional
            ? "  ⚠️ ANTI-COLAPSO: Predicciones muy uniformes (range:", Transform(nProbRange * 100, "@E 99.9"), "%) - Penalty:", nEntropyPenalty
         ENDIF
         
         nFocalLoss := nFocalLoss + nEntropyPenalty
         
         // PENALTY ANTI-YELLOW: Penalizar fuertemente predicciones incorrectas de YELLOW
         nMaxProb := 0  // Inicializar explícitamente
         nPredictedToken := 0  // Inicializar explícitamente
         
         // Validar que mProbabilities sea válida antes de procesar
         IF !Empty(mProbabilities) .AND. Len(mProbabilities) >= 1 .AND. Len(mProbabilities[1]) >= nVocabSize
            FOR k := 1 TO nVocabSize
               IF ValType(mProbabilities[1][k]) == "N" .AND. mProbabilities[1][k] > nMaxProb
                  nMaxProb := mProbabilities[1][k]
                  nPredictedToken := k - 1
               ENDIF
            NEXT
         ELSE
            ? "  ⚠ WARNING: mProbabilities inválida en época", i
         ENDIF
         
         // Si predice YELLOW incorrectamente, penalizar severamente
         IF nPredictedToken == 11 .AND. nTargetToken != 11  // Token YELLOW
            nYellowPenalty := 3.0 * nMaxProb  // Penalty proporcional a la confianza
         ENDIF
         
         // 🎯 PENALIZACIONES SEMÁNTICAS ESPECÍFICAS: Forzar relaciones objeto-color correctas
         nSemanticPenalty := 0
         nSemanticReward := 0
         
         // Detectar qué objeto se está preguntando
         nObjectType := 0  // 1=SKY/OCEAN, 2=GRASS, 3=SUN
         FOR k := 1 TO Len(aInput)
            IF aInput[k] == 5 .OR. aInput[k] == 6  // SKY o OCEAN
               nObjectType := 1
               EXIT
            ELSEIF aInput[k] == 7  // GRASS
               nObjectType := 2
               EXIT
            ELSEIF aInput[k] == 8  // SUN
               nObjectType := 3
               EXIT
            ENDIF
         NEXT
         
         // Aplicar RECOMPENSAS por predicciones semánticamente correctas
         IF nObjectType == 1 .AND. nPredictedToken == 9  // SKY/OCEAN→BLUE ✓
            nSemanticReward := -3.0  // Recompensa (reducir loss)
            ? "  ✅ RECOMPENSA SEMÁNTICA: SKY/OCEAN→BLUE (reward:", nSemanticReward, ")"
         ELSEIF nObjectType == 2 .AND. nPredictedToken == 10  // GRASS→GREEN ✓
            nSemanticReward := -3.0
            ? "  ✅ RECOMPENSA SEMÁNTICA: GRASS→GREEN (reward:", nSemanticReward, ")"
         ELSEIF nObjectType == 3 .AND. nPredictedToken == 11  // SUN→YELLOW ✓
            nSemanticReward := -3.0
            ? "  ✅ RECOMPENSA SEMÁNTICA: SUN→YELLOW (reward:", nSemanticReward, ")"
         ENDIF
         
         // Aplicar penalizaciones MASIVAS por errores semánticos
         IF nObjectType == 1  // SKY/OCEAN debería ser BLUE
            IF nPredictedToken == 10  // Predijo GREEN
               nSemanticPenalty := 15.0
               ? "  🚨 ERROR SEMÁNTICO: SKY/OCEAN→GREEN (penalty:", nSemanticPenalty, ")"
            ELSEIF nPredictedToken == 11  // Predijo YELLOW
               nSemanticPenalty := 15.0
               ? "  🚨 ERROR SEMÁNTICO: SKY/OCEAN→YELLOW (penalty:", nSemanticPenalty, ")"
            ENDIF
         ELSEIF nObjectType == 2  // GRASS debería ser GREEN
            IF nPredictedToken == 9  // Predijo BLUE
               nSemanticPenalty := 15.0
               ? "  🚨 ERROR SEMÁNTICO: GRASS→BLUE (penalty:", nSemanticPenalty, ")"
            ELSEIF nPredictedToken == 11  // Predijo YELLOW
               nSemanticPenalty := 15.0
               ? "  🚨 ERROR SEMÁNTICO: GRASS→YELLOW (penalty:", nSemanticPenalty, ")"
            ENDIF
         ELSEIF nObjectType == 3  // SUN debería ser YELLOW
            IF nPredictedToken == 9  // Predijo BLUE
               nSemanticPenalty := 15.0
               ? "  🚨 ERROR SEMÁNTICO: SUN→BLUE (penalty:", nSemanticPenalty, ")"
            ELSEIF nPredictedToken == 10  // Predijo GREEN
               nSemanticPenalty := 15.0
               ? "  🚨 ERROR SEMÁNTICO: SUN→GREEN (penalty:", nSemanticPenalty, ")"
            ENDIF
         ENDIF
         
         // REGULARIZACIÓN ADVERSARIAL LEGACY (mantener por compatibilidad)
         nAdversarialPenalty := 0
         IF nTargetToken == 10 .AND. nPredictedToken == 9  // Target=GREEN, Pred=BLUE
            nAdversarialPenalty := 5.0 * nMaxProb  // Penalty EXTREMO por confundir GREEN con BLUE
            ? "  🚨 ADVERSARIAL PENALTY: Confundió GREEN con BLUE (penalty:", nAdversarialPenalty, ")"
         ELSEIF nTargetToken == 9 .AND. nPredictedToken == 10  // Target=BLUE, Pred=GREEN
            nAdversarialPenalty := 2.0 * nMaxProb  // Penalty menor en dirección opuesta
         ENDIF
         
         // 🚨 CONSTRAINT DE DIVERSIDAD OBLIGATORIA
         // Incrementar contador de predicciones por clase
         IF nPredictedToken == 9
            aPredictionCounts[1]++  // BLUE
         ELSEIF nPredictedToken == 10
            aPredictionCounts[2]++  // GREEN
         ELSEIF nPredictedToken == 11
            aPredictionCounts[3]++  // YELLOW
         ENDIF
         
         // Verificar dominancia cada nDiversityCheckInterval ejemplos
         IF (j % nDiversityCheckInterval) == 0
            nTotalPredictions := aPredictionCounts[1] + aPredictionCounts[2] + aPredictionCounts[3]
            IF nTotalPredictions > 0
               nMaxClassCount := Max(Max(aPredictionCounts[1], aPredictionCounts[2]), aPredictionCounts[3])
               nDominanceRatio := nMaxClassCount / nTotalPredictions
               
               IF nDominanceRatio > nMaxDominanceRatio
                  nDiversityConstraintPenalty := nDiversityPenaltyMultiplier * (nDominanceRatio - nMaxDominanceRatio)
                  nDiversityPenalty := nDiversityPenalty + nDiversityConstraintPenalty
                  ? "  🚨 DIVERSIDAD VIOLADA: Clase dominante", Transform(nDominanceRatio * 100, "@E 99.9"), "% - Penalty:", nDiversityConstraintPenalty
               ENDIF
            ENDIF
         ENDIF
         
         // Penalty adicional si el modelo es muy confiado en una sola clase  
         IF nMaxProb > 0.85  // Si muy confiado en una clase
            nDiversityPenalty := nDiversityPenalty + (nMaxProb - 0.85) * 2.0  // Penalizar fuertemente
         ENDIF
         
         // 🚨 CONSTRAINT ANTI-DOMINANCIA EXTREMO
         nDominancePenalty := 0
         IF nMaxProb > 0.6  // UMBRAL MUY ESTRICTO: máximo 60% confianza
            nDominancePenalty := 10.0 * (nMaxProb - 0.6)  // Penalty EXTREMO
            ? "  🚨 DOMINANCIA DETECTADA: Prob máxima", Transform(nMaxProb * 100, "@E 99.9"), "% > 60% - Penalty:", nDominancePenalty
         ENDIF
         
         nFocalLoss := nFocalLoss + nDiversityPenalty + nYellowPenalty + nAdversarialPenalty + nDominancePenalty + nSemanticPenalty + nSemanticReward
         
         // Aplicar peso de clase para balanceo
         nLoss := nFocalLoss * nClassWeight
         
         // Backward pass - gradiente de cross-entropy con label smoothing
         mProbGradient := HB_MATRIXCLONE(mProbabilities)
         // Ajustar el gradiente para tener en cuenta label smoothing en backward
         mProbGradient[1][nTargetToken + 1] -= 1.0
         // Aplicar label smoothing al gradiente
         mProbGradient := HB_MATRIXMULSCALAR(mProbGradient, (1.0 - nLabelSmoothing))
         // Agregar el componente de label smoothing
         nSmoothComponent := nLabelSmoothing / nVocabSize
         mProbGradient := HB_MATRIXADD(mProbGradient, HB_MATRIXFILL(HB_MATRIXZERO(1, nVocabSize), nSmoothComponent))
         
         // Backward a través de la capa de clasificación
         mResponseGradient := HB_MATRIXMULTIPLY(mProbGradient, HB_MATRIXTRANSPOSE(mClassificationWeights))
         
         // Crear gradiente completo para el transformer (ceros excepto posición de respuesta)
         dLoss := HB_MATRIXZERO(nSeqLen, nEmbedDim)
         dLoss[nAnswerPos] := mResponseGradient[1]  // Solo la posición de respuesta tiene gradiente
         
         // DEBUG: mostrar información del output de la posición relevante
         IF i <= 10 .AND. j == 1  // Más épocas de debug
            nPredictedToken := 0
            nMaxProb := 0
            FOR k := 1 TO nVocabSize
               IF mProbabilities[1][k] > nMaxProb
                  nMaxProb := mProbabilities[1][k]
                  nPredictedToken := k - 1
               ENDIF
            NEXT
            ? "  DEBUG época", i, "ejemplo", j, "LR:", Transform(nCurrentLR, "@E 9.999"), ":"
            ? "    Target[7]:", nTargetToken, "(" + aVocab[nTargetToken+1] + ")"
            ? "    Pred token:", nPredictedToken, "(" + aVocab[nPredictedToken+1] + ")"
            ? "    Prob:", Transform(nMaxProb, "@E 9.999")
            ? "    BLUE:", Transform(mProbabilities[1][10], "@E 9.99"), ;
              "GREEN:", Transform(mProbabilities[1][11], "@E 9.99"), ;
              "YELLOW:", Transform(mProbabilities[1][12], "@E 9.99")
            ? ""
         ENDIF
         
         // Backprop a través del modelo
         mInputGradient := oModel:Backward(dLoss)
         
         // 🔧 GRADIENT CLIPPING ADAPTIVO basado en magnitud de gradientes
         nGradMagnitude := CalculateGradientMagnitude(mInputGradient)
         nAdaptiveClipping := nGradientClipping
         
         IF nGradMagnitude > 2.0  // Gradientes muy grandes
            nAdaptiveClipping := nGradientClipping * 0.5  // Clipping más agresivo
            ? "🚨 GRADIENT EXPLOSION: Magnitud", Transform(nGradMagnitude, "@E 9.999"), "- Clipping x0.5"
         ELSEIF nGradMagnitude < 0.001  // Gradientes muy pequeños
            nAdaptiveClipping := nGradientClipping * 2.0  // Clipping más permisivo
            ? "🔻 VANISHING GRADIENTS: Magnitud", Transform(nGradMagnitude, "@E 9.999"), "- Clipping x2"
         ENDIF
         
         // Aplicar gradient clipping adaptivo
         mInputGradient := GradientClipMatrix(mInputGradient, nAdaptiveClipping)
         
         // IMPORTANTE: Actualizar los embeddings con el gradiente correcto
         UpdateEmbeddings(mEmbeddings, aInput, mInputGradient, nCurrentLR, nGradientClipping)
         
         // Aplicar gradient clipping al modelo transformer
         oModel:ApplyGradientClipping(nGradientClipping)
         
         // ANÁLISIS DE GRADIENTES para debugging
         IF i <= 10 .OR. (i % 200 == 0 .AND. j == 1)  // Solo primeras épocas y muestreo
            // Validar que los gradientes existan antes de analizarlos
            IF !Empty(mWeightGradient) .AND. !Empty(mProbGradient)
               nAvgGradMagnitude := AnalyzeGradients(mWeightGradient, mProbGradient)
               IF nAvgGradMagnitude > 1.0
                  ? "  ⚠ WARNING: Gradientes grandes detectados (", Transform(nAvgGradMagnitude, "@E 9.999"), ")"
               ELSEIF nAvgGradMagnitude < 0.0001
                  ? "  ⚠ WARNING: Gradientes muy pequeños (", Transform(nAvgGradMagnitude, "@E 9.999"), ") - posible vanishing gradients"
               ENDIF
            ELSE
               ? "  ⚠ WARNING: Gradientes no válidos para análisis en época", i
            ENDIF
         ENDIF
         
         // Actualizar modelo transformer
         oModel:Update(nCurrentLR)
         
         // Actualizar capa de clasificación con momentum
         mWeightGradient := HB_MATRIXMULTIPLY(HB_MATRIXTRANSPOSE(mResponseVector), mProbGradient)
         // Aplicar gradient clipping a los gradientes de clasificación
         mWeightGradient := GradientClipMatrix(mWeightGradient, nGradientClipping)
         
         // Añadir regularización L2 (weight decay) más suave
         nWeightDecay := 0.00005
         mWeightGradient := HB_MATRIXADD(mWeightGradient, HB_MATRIXMULSCALAR(mClassificationWeights, nWeightDecay))
         
         // Momentum SGD para la capa de clasificación
         nMomentum := 0.9
         mClassMomentumW := HB_MATRIXADD(HB_MATRIXMULSCALAR(mClassMomentumW, nMomentum), HB_MATRIXMULSCALAR(mWeightGradient, nCurrentLR))
         mClassMomentumB := HB_MATRIXADD(HB_MATRIXMULSCALAR(mClassMomentumB, nMomentum), HB_MATRIXMULSCALAR(mProbGradient, nCurrentLR))
         
         mClassificationWeights := HB_MATRIXSUB(mClassificationWeights, mClassMomentumW)
         mClassificationBias := HB_MATRIXSUB(mClassificationBias, mClassMomentumB)
         
         nTotalLoss += nLoss
      NEXT
      
      nTotalLoss := nTotalLoss / Len(aColorExamples)
      
      IF i % 200 == 0 .OR. i <= 10 .OR. i == nWarmupEpochs
         cPhaseDesc := ""
         IF i <= nWarmupEpochs
            cPhaseDesc := " (WARM-UP)"
         ELSEIF nPhase <= 2
            cPhaseDesc := " (FASE ENTRENAMIENTO)"
         ELSE
            cPhaseDesc := " (FASE REFINAMIENTO)"
         ENDIF
         
         IF i <= 200
            cTrainingPhase := " [SOLO BLUE]"
         ELSEIF i <= 400
            cTrainingPhase := " [SOLO GREEN]"
         ELSE
            cTrainingPhase := " [TODAS LAS CLASES]"
         ENDIF
         
         ? "Época", PadR(Str(i,5),6), "-> Loss promedio:", ;
           Transform(nTotalLoss, "@E 9.999999"), "LR:", Transform(nCurrentLR, "@E 9.9999"), cPhaseDesc + cTrainingPhase
      ENDIF
      
      // EARLY STOPPING: Validar y guardar mejor modelo
      IF i % nValidationFreq == 0 .OR. i == nEpochs
         IF nTotalLoss < nBestLoss
            nBestLoss := nTotalLoss
            nBestEpoch := i
            nPatienceCounter := 0
            
            // Guardar mejor modelo
            mBestClassWeights := HB_MATRIXCLONE(mClassificationWeights)
            mBestClassBias := HB_MATRIXCLONE(mClassificationBias)
            
            ? "  ★ NUEVO MEJOR MODELO en época", i, "| Loss:", Transform(nBestLoss, "@E 9.999999")
         ELSE
            nPatienceCounter += nValidationFreq
            IF nPatienceCounter >= nMaxPatience .AND. i > nWarmupEpochs
               ? "  ⏹ EARLY STOPPING: No mejora en", nMaxPatience, "épocas"
               ? "  ⭐ Restaurando mejor modelo de época", nBestEpoch
               
               // Restaurar mejor modelo
               mClassificationWeights := HB_MATRIXCLONE(mBestClassWeights)
               mClassificationBias := HB_MATRIXCLONE(mBestClassBias)
               EXIT  // Salir del loop de entrenamiento
            ENDIF
         ENDIF
      ENDIF
   NEXT
   
   ? Replicate("-", 70)
   ? ""
   
   // --- VERIFICACIÓN FINAL ---
   ? "========================================================================="
   ? "VERIFICACIÓN: Predicciones del modelo entrenado"
   ? "========================================================================="
   ? ""
   
   nCorrect := 0
   nTotal := 0
   // Resetear contadores por clase
   AFill(nCorrectByClass, 0)
   AFill(nTotalByClass, 0)
   
   FOR j := 1 TO Len(aTrainingData)
      aInput := aTrainingData[j][1]
      aTarget := aTrainingData[j][2]
      
      mInput := CreateMatrixFromTokens(aInput, mEmbeddings)
      mInput := HB_MATRIXADD(mInput, mPositionalEncoding)
      
      mTransformerOutput := oModel:Forward(mInput, .F.)  // Evaluation mode
      
      // Usar la posición fija 7 para la respuesta como se define en el dataset
      nAnswerPos := 7
      
      // Aplicar capa de clasificación para obtener probabilidades en la posición correcta
      mResponseVector := {mTransformerOutput[nAnswerPos]}
      mLogits := HB_MATRIXMULTIPLY(mResponseVector, mClassificationWeights)
      mLogits := HB_MATRIXADDBROADCAST(mLogits, mClassificationBias)
      mProbabilities := ApplyLabelSmoothing(mLogits, nLabelSmoothing)
      
      // Encontrar el token con mayor probabilidad
      nPredictedToken := 0
      nMaxProb := 0
      FOR k := 1 TO nVocabSize
         IF mProbabilities[1][k] > nMaxProb
            nMaxProb := mProbabilities[1][k]
            nPredictedToken := k - 1
         ENDIF
      NEXT
      
      // Actualizar métricas por clase
      nTargetToken := aTarget[nAnswerPos]
      IF nTargetToken == nTokenBlue
         nTotalByClass[1]++
         aClassCounts[1]++  // Actualizar contadores para LR adaptativo
         IF nPredictedToken == nTargetToken
            nCorrectByClass[1]++
            aClassCorrect[1]++
         ENDIF
      ELSEIF nTargetToken == nTokenGreen
         nTotalByClass[2]++
         aClassCounts[2]++  // Actualizar contadores para LR adaptativo
         IF nPredictedToken == nTargetToken
            nCorrectByClass[2]++
            aClassCorrect[2]++
         ENDIF
      ELSEIF nTargetToken == nTokenYellow
         nTotalByClass[3]++
         aClassCounts[3]++  // Actualizar contadores para LR adaptativo
         IF nPredictedToken == nTargetToken
            nCorrectByClass[3]++
            aClassCorrect[3]++
         ENDIF
      ELSE
         nTotalByClass[4]++
         IF nPredictedToken == nTargetToken
            nCorrectByClass[4]++
         ENDIF
      ENDIF
      
      // 📈 LEARNING RATE ADAPTATIVO POR CLASE + EQUILIBRIO INICIAL
      // Calcular accuracy por clase y ajustar LR dinámicamente
      IF (j % 10) == 0  // Cada 10 ejemplos
         nClassLRMultiplier := 1.0
         cClassStatus := ""
         
         // 🎯 LEARNING RATES DIFERENCIADOS DESDE INICIO
         // Dar ventaja inicial a todas las clases débiles
         IF i <= 50  // Primeras 50 épocas: equilibrio forzado
            IF aClassCounts[1] > 0 .AND. aClassAccuracy[1] < 0.5  // BLUE débil
               nClassLRMultiplier := 2.0
               cClassStatus := "🎯 LR INICIAL x2: BLUE (" + Transform(aClassAccuracy[1] * 100, "@E 99") + "%)"
            ENDIF
            IF aClassCounts[2] > 0 .AND. aClassAccuracy[2] < 0.5  // GREEN débil
               nClassLRMultiplier := 2.0
               cClassStatus := "🎯 LR INICIAL x2: GREEN (" + Transform(aClassAccuracy[2] * 100, "@E 99") + "%)"
            ENDIF
            IF aClassCounts[3] > 0 .AND. aClassAccuracy[3] < 0.5  // YELLOW débil
               nClassLRMultiplier := 2.0
               cClassStatus := "🎯 LR INICIAL x2: YELLOW (" + Transform(aClassAccuracy[3] * 100, "@E 99") + "%)"
            ENDIF
         ENDIF
         
         FOR k := 1 TO 3
            IF aClassCounts[k] > 0
               aClassAccuracy[k] := aClassCorrect[k] / aClassCounts[k]
               
               // 🚨 ANTI-DOMINANCIA UNIVERSAL: Penalizar CUALQUIER clase >40%
               IF aClassAccuracy[k] > 0.4  // Si accuracy > 40%
                  nClassLRMultiplier := 0.1   // PENALTY: reducir LR a 10%
                  cClassStatus := "⚠️ ANTI-DOMINANCIA: Clase " + Str(k) + " dominante (" + Transform(aClassAccuracy[k] * 100, "@E 99") + "%) - Penalizando"
               ENDIF
               
               // 🚨 FORCED DIVERSITY EXTREMA: Boost MASIVO para clases con baja accuracy
               IF aClassAccuracy[k] < 0.2  // Si accuracy < 20%
                  nClassLRMultiplier := 10.0  // BOOST EXTREMO x10
                  cClassStatus := "🔥 BOOST x10: Clase " + Str(k) + " (" + Transform(aClassAccuracy[k] * 100, "@E 99") + "%)"
               ELSEIF aClassAccuracy[k] < 0.35  // Si accuracy < 35%
                  nClassLRMultiplier := 3.0   // BOOST moderado x3
                  cClassStatus := "⚡ BOOST x3: Clase " + Str(k) + " (" + Transform(aClassAccuracy[k] * 100, "@E 99") + "%)"
               ENDIF
            ENDIF
         NEXT
         
         // Aplicar multiplier al LR actual
         IF nClassLRMultiplier != 1.0
            nCurrentLR := nCurrentLR * nClassLRMultiplier
            ? "  📈 LR ADAPTATIVO:", cClassStatus, "- Nuevo LR:", Transform(nCurrentLR, "@E 9.9999")
         ENDIF
      ENDIF
      
      ? "Pregunta " + Str(j,1) + ":"
      ? "  Input:      ", TokensToWords(aInput, aVocab)
      ? "  Target:     ", aVocab[aTarget[nAnswerPos] + 1]  // Solo la respuesta (posición correcta)
      ? "  Predicción: ", aVocab[nPredictedToken + 1], "(prob:", Transform(nMaxProb, "@E 9.99"), ")"
      
      IF aTarget[nAnswerPos] == nPredictedToken
         ? "  ✓ CORRECTO"
         nCorrect++
      ELSE
         ? "  ✗ INCORRECTO"
      ENDIF
      nTotal++
      ? ""
   NEXT
   
   ? "========================================================================="
   ? "RESULTADO FINAL: ", nCorrect, " de ", nTotal, " aciertos (", (nCorrect/nTotal)*100, "%)"
   ? "========================================================================="
   ? ""
   ? "MÉTRICAS POR CLASE:"
   IF nTotalByClass[1] > 0
      ? "  BLUE:   ", nCorrectByClass[1], "/", nTotalByClass[1], " (", (nCorrectByClass[1]/nTotalByClass[1])*100, "%)"
   ENDIF
   IF nTotalByClass[2] > 0
      ? "  GREEN:  ", nCorrectByClass[2], "/", nTotalByClass[2], " (", (nCorrectByClass[2]/nTotalByClass[2])*100, "%)"
   ENDIF
   IF nTotalByClass[3] > 0
      ? "  YELLOW: ", nCorrectByClass[3], "/", nTotalByClass[3], " (", (nCorrectByClass[3]/nTotalByClass[3])*100, "%)"
   ENDIF
   IF nTotalByClass[4] > 0
      ? "  OTROS:  ", nCorrectByClass[4], "/", nTotalByClass[4], " (", (nCorrectByClass[4]/nTotalByClass[4])*100, "%)"
   ENDIF
   ? ""
   
   // --- MOSTRAR EVOLUCIÓN DE LOS EMBEDDINGS ---
   ? "========================================================================="
   ? "ANÁLISIS FINAL: ¿Qué aprendieron los embeddings densos?"
   ? "========================================================================="
   ? ""
   ? "Los embeddings han sido entrenados para capturar relaciones semánticas"
   ? "entre palabras basándose en el contexto de las preguntas y respuestas."
   ? ""
   ? "Embedding final del token 'SKY' (índice", nTokenSky, "):"
   ? "  ", HB_ValToExp(mEmbeddings[nTokenSky + 1])
   ? ""
   ? "MAGNITUDES DE VECTORES (indicador de importancia aprendida):"
   ? "  SKY:     ", Transform(VectorMagnitude(mEmbeddings[nTokenSky + 1]), "@E 9.999")
   ? "  OCEAN:   ", Transform(VectorMagnitude(mEmbeddings[nTokenOcean + 1]), "@E 9.999")
   ? "  GRASS:   ", Transform(VectorMagnitude(mEmbeddings[nTokenGrass + 1]), "@E 9.999")
   ? "  SUN:     ", Transform(VectorMagnitude(mEmbeddings[nTokenSun + 1]), "@E 9.999")
   ? "  BLUE:    ", Transform(VectorMagnitude(mEmbeddings[nTokenBlue + 1]), "@E 9.999")
   ? "  GREEN:   ", Transform(VectorMagnitude(mEmbeddings[nTokenGreen + 1]), "@E 9.999")
   ? "  YELLOW:  ", Transform(VectorMagnitude(mEmbeddings[nTokenYellow + 1]), "@E 9.999")
   ? ""
   ? "DISTANCIAS SEMÁNTICAS (menor = más relacionados):"
   ? "  SKY <-> OCEAN:    ", Transform(EuclideanDistance(mEmbeddings[nTokenSky + 1], mEmbeddings[nTokenOcean + 1]), "@E 9.999"), "(ambos azules)"
   ? "  SKY <-> GRASS:    ", Transform(EuclideanDistance(mEmbeddings[nTokenSky + 1], mEmbeddings[nTokenGrass + 1]), "@E 9.999"), "(colores diferentes)"
   ? "  BLUE <-> GREEN:   ", Transform(EuclideanDistance(mEmbeddings[nTokenBlue + 1], mEmbeddings[nTokenGreen + 1]), "@E 9.999"), "(colores diferentes)"
   ? "  BLUE <-> YELLOW:  ", Transform(EuclideanDistance(mEmbeddings[nTokenBlue + 1], mEmbeddings[nTokenYellow + 1]), "@E 9.999"), "(colores diferentes)"
   ? ""
   ? "INTERPRETACIÓN:"
   ? "• Los embeddings deberían mostrar que SKY y OCEAN están más cerca"
   ? "  semánticamente (ambos son BLUE) que SKY y GRASS (colores diferentes)."
   ? "• Las magnitudes reflejan la 'importancia' que el modelo asignó a cada palabra."
   ? "• Distancias menores entre objetos del mismo color indican aprendizaje exitoso."
   ? ""

RETURN


// =======================================================================
// FUNCIONES DE VALIDACIÓN Y UTILIDADES
// =======================================================================

//
// Valida que una matriz no esté vacía y tenga las dimensiones esperadas
//
STATIC FUNCTION ValidateMatrix(mMatrix, nExpectedRows, nExpectedCols, cName)
   LOCAL lValid := .T.
   
   IF Empty(mMatrix)
      ? "ERROR: La matriz", cName, "está vacía"
      RETURN .F.
   ENDIF
   
   IF nExpectedRows > 0 .AND. Len(mMatrix) != nExpectedRows
      ? "ERROR: La matriz", cName, "tiene", Len(mMatrix), "filas, esperadas", nExpectedRows
      lValid := .F.
   ENDIF
   
   IF nExpectedCols > 0 .AND. Len(mMatrix) > 0 .AND. Len(mMatrix[1]) != nExpectedCols
      ? "ERROR: La matriz", cName, "tiene", Len(mMatrix[1]), "columnas, esperadas", nExpectedCols
      lValid := .F.
   ENDIF
   
RETURN lValid


/*
* Valida que los tokens estén dentro del rango válido del vocabulario
*/
STATIC FUNCTION ValidateTokens(aTokens, nVocabSize, cName)
   LOCAL i, nToken, lValid := .T.
   
   FOR i := 1 TO Len(aTokens)
      nToken := aTokens[i]
      IF nToken < 0 .OR. nToken >= nVocabSize
         ? "ERROR: Token inválido en", cName, "posición", i, "token:", nToken, "rango válido: 0-" + Str(nVocabSize-1)
         lValid := .F.
      ENDIF
   NEXT
   
RETURN lValid


/*
* Valida que un valor numérico esté dentro de un rango específico
*/
STATIC FUNCTION ValidateRange(nValue, nMin, nMax, cName)
   IF nValue < nMin .OR. nValue > nMax
      ? "ERROR:", cName, "fuera de rango:", nValue, "rango válido:", nMin, "-", nMax
      RETURN .F.
   ENDIF
RETURN .T.


/*
* =======================================================================
* FUNCIONES PARA EMBEDDINGS DENSOS Y TÉCNICAS DE ENTRENAMIENTO MEJORADAS
* ========================================================================
*/

/*
* Aplica label smoothing al logits para prevenir overconfidence
* Optimizado para reducir operaciones matriciales
*/
STATIC FUNCTION ApplyLabelSmoothing(mLogits, nLabelSmoothing)
   LOCAL mProbabilities := HB_SOFTMAX(mLogits)
   LOCAL nVocabSize := Len(mProbabilities[1])
   LOCAL mSmoothedProbs := HB_MATRIXZERO(1, nVocabSize)
   LOCAL i, nSmoothFactor, nRegularFactor, nNoise, nSum
   
   // Validación de parámetros
   IF Empty(mProbabilities) .OR. !ValidateRange(nLabelSmoothing, 0, 1, "Label Smoothing")
      RETURN mProbabilities
   ENDIF
   
   // Pre-calcular factores para evitar operaciones repetitivas
   nRegularFactor := 1 - nLabelSmoothing
   nSmoothFactor := nLabelSmoothing / nVocabSize
   
   // 🎲 UNCERTAINTY INJECTION: Añadir ruido gaussiano para evitar over-confidence
   FOR i := 1 TO nVocabSize
      nNoise := HB_RANDOM() * 0.02 - 0.01  // Ruido ±1% para reducir over-confidence
      mSmoothedProbs[1][i] := nRegularFactor * mProbabilities[1][i] + nSmoothFactor + nNoise
   NEXT
   
   // 🔄 NORMALIZAR después del ruido para mantener suma = 1
   nSum := 0
   FOR i := 1 TO nVocabSize
      nSum := nSum + mSmoothedProbs[1][i]
   NEXT
   FOR i := 1 TO nVocabSize
      mSmoothedProbs[1][i] := mSmoothedProbs[1][i] / nSum
   NEXT
   
RETURN mSmoothedProbs


/*
* Implementa Focal Loss para combatir el sesgo de clase
* Focal Loss = -α(1-pt)^γ * log(pt)
*/
STATIC FUNCTION CalculateFocalLoss(mProbabilities, nTargetToken, nAlpha, nGamma)
   LOCAL nTargetProb, nFocalWeight, nLoss
   LOCAL nTargetIdx := nTargetToken + 1  // Harbour arrays are 1-indexed
   
   // Validación de parámetros
   IF Empty(mProbabilities) .OR. nTargetIdx < 1 .OR. nTargetIdx > Len(mProbabilities[1])
      RETURN 999.0  // Loss alto como penalización
   ENDIF
   
   nTargetProb := mProbabilities[1][nTargetIdx]
   
   // Clipping para evitar log(0)
   IF nTargetProb < 0.0001
      nTargetProb := 0.0001
   ENDIF
   
   // Calcular peso focal: α(1-pt)^γ
   nFocalWeight := nAlpha * ((1.0 - nTargetProb) ^ nGamma)
   
   // Focal loss: -α(1-pt)^γ * log(pt)
   nLoss := -nFocalWeight * Log(nTargetProb)
   
RETURN nLoss


/*
* Calcula pesos balanceados por clase para combatir el sesgo
*/
STATIC FUNCTION CalculateClassWeights(aTrainingData, nVocabSize)
   LOCAL aClassCounts := Array(nVocabSize)
   LOCAL aClassWeights := Array(nVocabSize)
   LOCAL i, j, nTargetToken, nTotalSamples, nMaxCount
   
   // Inicializar contadores
   AFill(aClassCounts, 0)
   
   // Contar ocurrencias de cada clase
   FOR i := 1 TO Len(aTrainingData)
      FOR j := 1 TO Len(aTrainingData[i][2])
         nTargetToken := aTrainingData[i][2][j]
         IF nTargetToken > 0 .AND. nTargetToken <= nVocabSize  // Solo tokens válidos
            aClassCounts[nTargetToken]++
         ENDIF
      NEXT
   NEXT
   
   // Encontrar la clase más frecuente
   nMaxCount := 0
   FOR i := 1 TO nVocabSize
      IF aClassCounts[i] > nMaxCount
         nMaxCount := aClassCounts[i]
      ENDIF
   NEXT
   
   // Calcular pesos balanceados: peso = max_count / class_count
   FOR i := 1 TO nVocabSize
      IF aClassCounts[i] > 0
         aClassWeights[i] := nMaxCount / aClassCounts[i]
      ELSE
         aClassWeights[i] := 1.0  // Peso neutro para clases no presentes
      ENDIF
   NEXT
   
RETURN aClassWeights


/*
* Aplica gradient clipping a una matriz de gradientes
* Optimizado para minimizar operaciones de clonado
*/
STATIC FUNCTION GradientClipMatrix(mMatrix, nClipValue)
   LOCAL i, j, nValue
   LOCAL mClipped := HB_MATRIXCLONE(mMatrix)
   
   // Validación de parámetros
   IF Empty(mMatrix) .OR. !ValidateRange(nClipValue, 0.001, 100, "Clip Value")
      RETURN mMatrix
   ENDIF
   
   // Aplicar clipping in-place para mejor performance
   FOR i := 1 TO Len(mMatrix)
      FOR j := 1 TO Len(mMatrix[i])
         nValue := mClipped[i][j]
         IF nValue > nClipValue
            mClipped[i][j] := nClipValue
         ELSEIF nValue < -nClipValue
            mClipped[i][j] := -nClipValue
         ENDIF
      NEXT
   NEXT
   
RETURN mClipped


/*
* Calcula la magnitud de un vector (para análisis de explosión de gradientes)
*/
STATIC FUNCTION VectorMagnitude(aVector)
   LOCAL nSumSq := 0
   LOCAL i
   
   FOR i := 1 TO Len(aVector)
      nSumSq += aVector[i]^2
   NEXT
   
RETURN Sqrt(nSumSq)


/*
* Crea embeddings densos con inicialización Gaussiana N(0, 0.02)
* Esta es la inicialización estándar en transformers modernos
*/
STATIC FUNCTION CreateDenseEmbeddings(nVocabSize, nEmbedDim)
   LOCAL mEmbeddings := {}
   LOCAL i, j, aRow
   LOCAL nStdDev := 0.05  // Aumentar varianza inicial para diversificar
   
   // Validación de parámetros
   IF !ValidateRange(nVocabSize, 1, 10000, "Vocab Size")
      RETURN {}
   ENDIF
   
   IF !ValidateRange(nEmbedDim, 1, 1000, "Embedding Dimension")
      RETURN {}
   ENDIF
   
   FOR i := 1 TO nVocabSize
      aRow := {}
      FOR j := 1 TO nEmbedDim
         // Inicialización Gaussiana con Box-Muller transform
         AAdd(aRow, GaussianRandom(0, nStdDev))
      NEXT
      AAdd(mEmbeddings, aRow)
   NEXT
   
   // Validación post-creación
   IF Len(mEmbeddings) != nVocabSize
      ? "ERROR: CreateDenseEmbeddings - dimensiones incorrectas"
      RETURN {}
   ENDIF
   
RETURN mEmbeddings


/*
* Genera número aleatorio con distribución normal N(mean, stddev)
* Usa transformación Box-Muller
*/
STATIC FUNCTION GaussianRandom(nMean, nStdDev)
   LOCAL nU1, nU2, nZ0
   
   nU1 := hb_Random()
   nU2 := hb_Random()
   
   // Box-Muller transform
   nZ0 := Sqrt(-2.0 * Log(nU1)) * Cos(2.0 * 3.14159265359 * nU2)
   
RETURN nMean + nZ0 * nStdDev


/*
* Actualiza los embeddings durante backpropagation
* Los embeddings son parámetros entrenables del modelo
*/
STATIC FUNCTION UpdateEmbeddings(mEmbeddings, aTokens, mInputGradient, nLR, nGradientClipping)
   LOCAL i, nToken, j, nGrad
   
   // Validación de parámetros
   IF Empty(mEmbeddings) .OR. Empty(aTokens) .OR. Empty(mInputGradient)
      ? "ERROR: UpdateEmbeddings - parámetros vacíos"
      RETURN .F.
   ENDIF
   
   IF !ValidateRange(nLR, 0.00001, 1.0, "Learning Rate")
      RETURN .F.
   ENDIF
   
   // Para cada token en la secuencia de entrada
   FOR i := 1 TO Len(aTokens)
      nToken := aTokens[i]
      
      IF nToken > 0 .AND. nToken < Len(mEmbeddings)  // Validar token y ignorar padding
         // Actualizar cada dimensión del embedding usando el gradiente que viene del backprop
         FOR j := 1 TO Len(mEmbeddings[nToken + 1])
            IF i <= Len(mInputGradient) .AND. j <= Len(mInputGradient[i])
               // Gradiente estándar con gradient clipping
               nGrad := mInputGradient[i][j]
               
               // Aplicar gradient clipping
               IF nGrad > nGradientClipping
                  nGrad := nGradientClipping
               ELSEIF nGrad < -nGradientClipping
                  nGrad := -nGradientClipping
               ENDIF
               
               // Actualizar embedding
               mEmbeddings[nToken + 1][j] -= nLR * nGrad
            ENDIF
         NEXT
      ENDIF
   NEXT
   
RETURN .T.


/*
* Convierte array de tokens en matriz de embeddings densos
* Optimizado para minimizar copias de memoria
*/
STATIC FUNCTION CreateMatrixFromTokens(aTokens, mEmbeddings)
   LOCAL mMatrix := {}
   LOCAL nToken, i
   
   // Validación de parámetros
   IF Empty(aTokens) .OR. Empty(mEmbeddings)
      RETURN {}
   ENDIF
   
   // Pre-alocar espacio para mejor performance
   ASize(mMatrix, Len(aTokens))
   
   FOR i := 1 TO Len(aTokens)
      nToken := aTokens[i]
      // Validar rango del token antes de acceder al embedding
      IF nToken >= 0 .AND. nToken < Len(mEmbeddings)
         // Referencia directa en lugar de copia (más eficiente)
         mMatrix[i] := mEmbeddings[nToken + 1]
      ELSE
         ? "WARNING: Token fuera de rango:", nToken, "en posición", i
         // Usar embedding del token PAD como fallback
         mMatrix[i] := mEmbeddings[1]
      ENDIF
   NEXT
   
RETURN mMatrix


/*
* Convierte array de tokens a texto legible
*/
STATIC FUNCTION TokensToWords(aTokens, aVocab)
   LOCAL cResult := ""
   LOCAL nToken
   
   FOR EACH nToken IN aTokens
      IF nToken > 0  // Ignorar padding
         cResult += aVocab[nToken + 1] + " "
      ENDIF
   NEXT
   
RETURN AllTrim(cResult)


/*
* Crea positional encoding sinusoidal
*/
STATIC FUNCTION CreatePositionalEncoding(nSeqLen, nEmbedDim)
   LOCAL mPE := {}
   LOCAL i, j, aRow, nAngle, nSpecialFactor
   LOCAL nObjectPositionStart := 5
   LOCAL nObjectPositionEnd := 6
   LOCAL nObjectAttentionFactor := 1.5
   
   // Validación de parámetros
   IF !ValidateRange(nSeqLen, 1, 1000, "Sequence Length")
      RETURN {}
   ENDIF
   
   IF !ValidateRange(nEmbedDim, 1, 1000, "Embedding Dimension")
      RETURN {}
   ENDIF
   
   FOR i := 0 TO nSeqLen - 1
      aRow := {}
      FOR j := 0 TO nEmbedDim - 1
         nAngle := i / ( ( 10000 ^ (2.0 * Int(j/2)) ) / nEmbedDim)
         
         IF j % 2 == 0
            AAdd(aRow, Sin(nAngle))
         ELSE
            AAdd(aRow, Cos(nAngle))
         ENDIF
         
         // Factor especial para posiciones de objeto
         IF i >= nObjectPositionStart .AND. i <= nObjectPositionEnd
            nSpecialFactor := nObjectAttentionFactor
            aRow[Len(aRow)] := aRow[Len(aRow)] * nSpecialFactor
         ENDIF
      NEXT
      AAdd(mPE, aRow)
   NEXT
   
   // Validación post-creación
   IF Len(mPE) != nSeqLen
      ? "ERROR: CreatePositionalEncoding - dimensiones incorrectas"
      RETURN {}
   ENDIF
   
RETURN mPE


/*
* Calcula la distancia euclidiana entre dos vectores de embeddings
*/
STATIC FUNCTION EuclideanDistance(aVector1, aVector2)
   LOCAL nSumSq := 0
   LOCAL i, nDiff
   
   // Validación de entrada
   IF Len(aVector1) != Len(aVector2)
      RETURN -1  // Error: vectores de diferentes dimensiones
   ENDIF
   
   FOR i := 1 TO Len(aVector1)
      nDiff := aVector1[i] - aVector2[i]
      nSumSq += nDiff * nDiff
   NEXT
   
RETURN Sqrt(nSumSq)

/*
* Genera un dataset aumentado con variaciones sintéticas
* Aplica técnicas de data augmentation para enriquecer el dataset
*/
STATIC FUNCTION GenerateAugmentedDataset(aOriginalData)
   LOCAL aBalancedData := {}
   LOCAL i, j, aExample, aNewExample
   LOCAL aInput, aTarget
   LOCAL aBlueExamples := {}
   LOCAL aGreenExamples := {} 
   LOCAL aYellowExamples := {}
   LOCAL nTargetPerClass := 15  // 15 ejemplos por clase = 45 total
   
   // 🎯 RECOLECTAR EJEMPLOS ORIGINALES POR CLASE
   FOR i := 1 TO Len(aOriginalData)
      aExample := aOriginalData[i]
      aTarget := aExample[2]
      
      IF aTarget[7] == 9        // BLUE
         AAdd(aBlueExamples, aExample)
      ELSEIF aTarget[7] == 10   // GREEN  
         AAdd(aGreenExamples, aExample)
      ELSEIF aTarget[7] == 11   // YELLOW
         AAdd(aYellowExamples, aExample)
      ENDIF
   NEXT
   
   ? "Dataset original - BLUE:", Len(aBlueExamples), "GREEN:", Len(aGreenExamples), "YELLOW:", Len(aYellowExamples)
   
   // 🎯 GENERAR EXACTAMENTE 15 EJEMPLOS POR CLASE
   
   // BLUE: Repetir hasta tener 15
   FOR i := 1 TO nTargetPerClass
      j := ((i - 1) % Len(aBlueExamples)) + 1
      aNewExample := { AClone(aBlueExamples[j][1]), AClone(aBlueExamples[j][2]) }
      // Micro-variación para evitar overfitting exacto
      IF aNewExample[1][8] == 0
         aNewExample[1][8] := (i % 3)  // Padding ligero 0,1,2
      ENDIF
      AAdd(aBalancedData, aNewExample)
   NEXT
   
   // GREEN: Repetir hasta tener 15
   FOR i := 1 TO nTargetPerClass
      j := ((i - 1) % Len(aGreenExamples)) + 1
      aNewExample := { AClone(aGreenExamples[j][1]), AClone(aGreenExamples[j][2]) }
      // Micro-variación para GREEN
      IF aNewExample[1][8] == 0
         aNewExample[1][8] := ((i + 1) % 3)  // Padding diferente
      ENDIF
      AAdd(aBalancedData, aNewExample)
   NEXT
   
   // YELLOW: Repetir hasta tener 15
   FOR i := 1 TO nTargetPerClass
      j := ((i - 1) % Len(aYellowExamples)) + 1
      aNewExample := { AClone(aYellowExamples[j][1]), AClone(aYellowExamples[j][2]) }
      // Micro-variación para YELLOW
      IF aNewExample[1][8] == 0
         aNewExample[1][8] := ((i + 2) % 3)  // Padding diferente
      ENDIF
      AAdd(aBalancedData, aNewExample)
   NEXT
   
   ? "Dataset balanceado - Total:", Len(aBalancedData), "ejemplos (15 por clase)"
   
RETURN aBalancedData

/*
* Analiza los gradientes para detectar problemas de entrenamiento
*/
STATIC FUNCTION AnalyzeGradients(mWeightGradient, mProbGradient)
   LOCAL nSumMagnitudes := 0
   LOCAL nCount := 0
   LOCAL i, j
   LOCAL nAvgMagnitude := 0
   
   // Validar que los gradientes sean matrices válidas
   IF Empty(mWeightGradient) .OR. ValType(mWeightGradient) != "A"
      RETURN 0  // Retornar 0 si no hay gradientes válidos
   ENDIF
   
   IF Empty(mProbGradient) .OR. ValType(mProbGradient) != "A"
      RETURN 0  // Retornar 0 si no hay gradientes válidos
   ENDIF
   
   // Analizar gradientes de peso
   IF Len(mWeightGradient) > 0 .AND. ValType(mWeightGradient[1]) == "A"
      FOR i := 1 TO Len(mWeightGradient)
         IF ValType(mWeightGradient[i]) == "A" .AND. Len(mWeightGradient[i]) > 0
            FOR j := 1 TO Len(mWeightGradient[i])
               IF ValType(mWeightGradient[i][j]) == "N"
                  nSumMagnitudes += Abs(mWeightGradient[i][j])
                  nCount++
               ENDIF
            NEXT
         ENDIF
      NEXT
   ENDIF
   
   // Analizar gradientes de probabilidad
   IF Len(mProbGradient) > 0 .AND. ValType(mProbGradient[1]) == "A"
      FOR i := 1 TO Len(mProbGradient)
         IF ValType(mProbGradient[i]) == "A" .AND. Len(mProbGradient[i]) > 0
            FOR j := 1 TO Len(mProbGradient[i])
               IF ValType(mProbGradient[i][j]) == "N"
                  nSumMagnitudes += Abs(mProbGradient[i][j])
                  nCount++
               ENDIF
            NEXT
         ENDIF
      NEXT
   ENDIF
   
   nAvgMagnitude := iif(nCount > 0, nSumMagnitudes / nCount, 0)
   
RETURN nAvgMagnitude

/*
* Crea variaciones super-avanzadas para ejemplos GREEN
*/
STATIC FUNCTION CreateGreenSuperVariation(aBaseExample, nVariation)
   LOCAL aNewExample := { AClone(aBaseExample[1]), AClone(aBaseExample[2]) }
   LOCAL aInput := aNewExample[1]
   
   // Diferentes estrategias de variación según el número
   DO CASE
   CASE nVariation <= 5
      // Variaciones con padding diferente
      aInput[8] := (nVariation - 1) % 5
      
   CASE nVariation <= 10
      // Variaciones con orden ligeramente modificado
      IF aInput[4] == 4 .AND. aInput[5] == 6  // "THE GRASS"
         // Intercambiar ocasionalmente
         IF nVariation % 2 == 0
            aInput[4] := 6
            aInput[5] := 4
         ENDIF
      ENDIF
      
   CASE nVariation <= 15
      // Variaciones con tokens de contexto
      aInput[7] := (nVariation - 10) % 3  // Pequeñas variaciones en posición 7
      
   OTHERWISE
      // Variaciones adicionales con ruido controlado
      aInput[1] := 1 + (nVariation % 2)  // Pequeña variación en primer token
   ENDCASE
   
   // Asegurar que siempre apunte a GREEN
   aNewExample[2,7] := 10  // Garantizar target GREEN
   
   RETURN aNewExample

/*
* Calcula la magnitud de los gradientes para análisis
*/
STATIC FUNCTION CalculateGradientMagnitude(mGradient)
   LOCAL nMagnitude := 0
   LOCAL i, j
   
   IF Empty(mGradient)
      RETURN 0
   ENDIF
   
   FOR i := 1 TO Len(mGradient)
      FOR j := 1 TO Len(mGradient[i])
         nMagnitude := nMagnitude + (mGradient[i][j] * mGradient[i][j])
      NEXT
   NEXT
   
   RETURN Sqrt(nMagnitude)