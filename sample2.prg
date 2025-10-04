/*
* ========================================================================
* EJEMPLO DIDÁCTICO: EMBEDDINGS DENSOS vs       // Learning rate scheduling: reducir cada 600 epochs
      nCurrentLR := nLearningRate
      IF i > 600
         nCurrentLR := nLearningRate * 0.8
      ENDIF
      IF i > 1200
         nCurrentLR := nLearningRate * 0.5
      ENDIF
      IF i > 1500
         nCurrentLR := nLearningRate * 0.3
      ENDIF
      
      // Entrenamiento por fases: enfocarse en diferentes colores
      LOCAL nPhase := Int((i-1) / 600) + 1  // Fase 1, 2, 3
      LOCAL aColorExamples := {}
      
      IF nPhase == 1
         // Fase 1: Enfoque en GREEN (primeras 600 epochs)
         FOR j := 1 TO Len(aTrainingData)
            IF aTrainingData[j][2][7] == 10  // GREEN
               AAdd(aColorExamples, aTrainingData[j])
            ENDIF
         NEXT
         // Añadir algunos ejemplos de otros colores para diversidad
         FOR j := 1 TO Len(aTrainingData)
            IF aTrainingData[j][2][7] != 10 .AND. Len(aColorExamples) < 9
               AAdd(aColorExamples, aTrainingData[j])
            ENDIF
         NEXT
      ELSEIF nPhase == 2
         // Fase 2: Enfoque en BLUE (epochs 601-1200)
         FOR j := 1 TO Len(aTrainingData)
            IF aTrainingData[j][2][7] == 9  // BLUE
               AAdd(aColorExamples, aTrainingData[j])
            ENDIF
         NEXT
         FOR j := 1 TO Len(aTrainingData)
            IF aTrainingData[j][2][7] != 9 .AND. Len(aColorExamples) < 9
               AAdd(aColorExamples, aTrainingData[j])
            ENDIF
         NEXT
      ELSE
         // Fase 3: Todos los colores balanceados (epochs 1201+)
         aColorExamples := aTrainingData
      ENDIF
      
      // Validación de seguridad: asegurar que siempre tenemos ejemplos
      IF Len(aColorExamples) == 0
         aColorExamples := aTrainingData
      ENDIFarea: Question Answering sobre colores de objetos
* Vocabulario: 15 tokens (objetos, colores, palabras de pregunta)
* Embeddings: Densos de dimensión 8 (aprenden relaciones semánticas)
* ========================================================================
*/

#include "hbclass.ch"

REQUEST HB_GT_STD, HB_GT_STD_DEFAULT

PROCEDURE Main()
   LOCAL nVocabSize := 15
   LOCAL nEmbedDim := 8        // Embedding denso pequeño pero efectivo
   LOCAL nHiddenDim := 32
   LOCAL nHeadDim := 8
   LOCAL nLayers := 2
   LOCAL nEpochs := 1800        // Más epochs con enfoque por colores
   LOCAL nLearningRate := 0.001  // Learning rate más alto para mejor aprendizaje
   LOCAL oModel, i, j, k, l
   LOCAL mEmbeddings, mPositionalEncoding
   LOCAL aInput, aTarget, mInput, mTarget, mOutput
   LOCAL nTotalLoss, nLoss, dLoss
   LOCAL aPredictedTokens, mInputGradient
   LOCAL mClassificationWeights, mClassificationBias
   LOCAL mTransformerOutput, mResponseVector, mLogits, mProbabilities
   LOCAL nTargetToken, mProbGradient, mResponseGradient, mWeightGradient
   LOCAL nPredictedToken, nMaxProb, nXavierStd, nCurrentLR, nWeightDecay
   LOCAL nEntropyRegularization, nProb, nMomentum, nPhase, cPhaseDesc
   LOCAL mClassMomentumW, mClassMomentumB  // Momentum para clasificación
   LOCAL aColorExamples
   
   // --- VOCABULARIO EXPANDIDO (15 tokens) ---
   // Índices: 0=PAD, 1=WHAT, 2=COLOR, 3=IS, 4=THE, 
   //          5=SKY, 6=GRASS, 7=SUN, 8=OCEAN,
   //          9=BLUE, 10=GREEN, 11=YELLOW, 12=RED, 13=ORANGE, 14=?
   LOCAL aVocab := { ;
      "PAD", "WHAT", "COLOR", "IS", "THE", ;
      "SKY", "GRASS", "SUN", "OCEAN", ;
      "BLUE", "GREEN", "YELLOW", "RED", "ORANGE", "?" ;
   }
   
   // --- DATASET ESTRATÉGICAMENTE BALANCEADO (18 ejemplos) ---
   // Alternando colores para evitar sobreajuste hacia un solo color
   LOCAL aTrainingData := { ;
      { {1,2,3,4,6,14,0,0}, {0,0,0,0,0,0,10,0} }, ; // "WHAT COLOR IS THE GRASS?" -> "GREEN" (empezar con GREEN)
      { {1,2,3,4,5,14,0,0}, {0,0,0,0,0,0,9,0} },  ; // "WHAT COLOR IS THE SKY?" -> "BLUE"
      { {1,2,3,4,7,14,0,0}, {0,0,0,0,0,0,11,0} }, ; // "WHAT COLOR IS THE SUN?" -> "YELLOW"
      { {1,2,3,4,6,14,0,0}, {0,0,0,0,0,0,10,0} }, ; // "WHAT COLOR IS THE GRASS?" -> "GREEN"
      { {1,2,3,4,8,14,0,0}, {0,0,0,0,0,0,9,0} },  ; // "WHAT COLOR IS THE OCEAN?" -> "BLUE"
      { {1,2,3,4,7,14,0,0}, {0,0,0,0,0,0,11,0} }, ; // "WHAT COLOR IS THE SUN?" -> "YELLOW"
      { {1,2,3,4,6,14,0,0}, {0,0,0,0,0,0,10,0} }, ; // "WHAT COLOR IS THE GRASS?" -> "GREEN"
      { {1,2,3,4,5,14,0,0}, {0,0,0,0,0,0,9,0} },  ; // "WHAT COLOR IS THE SKY?" -> "BLUE"
      { {1,2,3,4,7,14,0,0}, {0,0,0,0,0,0,11,0} }, ; // "WHAT COLOR IS THE SUN?" -> "YELLOW"
      { {1,2,3,4,6,14,0,0}, {0,0,0,0,0,0,10,0} }, ; // "WHAT COLOR IS THE GRASS?" -> "GREEN"
      { {1,2,3,4,8,14,0,0}, {0,0,0,0,0,0,9,0} },  ; // "WHAT COLOR IS THE OCEAN?" -> "BLUE"
      { {1,2,3,4,7,14,0,0}, {0,0,0,0,0,0,11,0} }, ; // "WHAT COLOR IS THE SUN?" -> "YELLOW"
      { {1,2,3,4,6,14,0,0}, {0,0,0,0,0,0,10,0} }, ; // "WHAT COLOR IS THE GRASS?" -> "GREEN"
      { {1,2,3,4,5,14,0,0}, {0,0,0,0,0,0,9,0} },  ; // "WHAT COLOR IS THE SKY?" -> "BLUE"
      { {1,2,3,4,7,14,0,0}, {0,0,0,0,0,0,11,0} }, ; // "WHAT COLOR IS THE SUN?" -> "YELLOW"
      { {1,2,3,4,6,14,0,0}, {0,0,0,0,0,0,10,0} }, ; // "WHAT COLOR IS THE GRASS?" -> "GREEN"
      { {1,2,3,4,8,14,0,0}, {0,0,0,0,0,0,9,0} },  ; // "WHAT COLOR IS THE OCEAN?" -> "BLUE"
      { {1,2,3,4,7,14,0,0}, {0,0,0,0,0,0,11,0} }  ; // "WHAT COLOR IS THE SUN?" -> "YELLOW"
   }
   
   LOCAL nSeqLen := 8
   
   ? "========================================================================="
   ? "TRANSFORMER CON EMBEDDINGS DENSOS ENTRENABLES"
   ? "========================================================================="
   ? ""
   ? "VOCABULARIO (", nVocabSize, "tokens ):"
   ? HB_ValToExp(aVocab)
   ? ""
   ? "VENTAJAS DE EMBEDDINGS DENSOS:"
   ? "  1. Capturan relaciones semánticas (ej: 'SKY' y 'OCEAN' están cerca)"
   ? "  2. Dimensión reducida:", nEmbedDim, "vs", nVocabSize, "(one-hot)"
   ? "  3. Se entrenan con el modelo (aprenden significado del contexto)"
   ? "  4. Permiten generalización (palabras similares -> vectores similares)"
   ? ""
   ? "========================================================================="
   ? ""
   
   // --- CREAR EMBEDDINGS DENSOS (inicialización Gaussiana) ---
   mEmbeddings := CreateDenseEmbeddings(nVocabSize, nEmbedDim)
   
   ? "Embeddings inicializados con distribución N(0, 0.02)"
   ? "Ejemplo - Embedding del token 'SKY' (índice 5):"
   ? "  ", HB_ValToExp(mEmbeddings[6])  // +1 porque arrays en Harbour son base-1
   ? ""
   
   // --- CREAR POSITIONAL ENCODING ---
   mPositionalEncoding := CreatePositionalEncoding(nSeqLen, nEmbedDim)
   
   // --- CREAR MODELO CON CAPA DE CLASIFICACIÓN ---
   oModel := TransformerModel():New(nLayers, nEmbedDim, nHiddenDim, nHeadDim)
   
   // Crear capa de clasificación con mejor inicialización (Xavier/Glorot)
   nXavierStd := Sqrt(2.0 / (nEmbedDim + nVocabSize))
   mClassificationWeights := CreateDenseEmbeddings(nEmbedDim, nVocabSize)
   // Reescalar con Xavier initialization
   FOR i := 1 TO nEmbedDim
      FOR j := 1 TO nVocabSize
         mClassificationWeights[i][j] *= nXavierStd
      NEXT
   NEXT
   mClassificationBias := HB_MATRIXZERO(1, nVocabSize)
   
   // Inicializar momentum para capa de clasificación
   mClassMomentumW := HB_MATRIXZERO(nEmbedDim, nVocabSize)
   mClassMomentumB := HB_MATRIXZERO(1, nVocabSize)
   
   ? "Iniciando entrenamiento..."
   ? Replicate("-", 70)
   
   // --- BUCLE DE ENTRENAMIENTO SIMPLIFICADO ---
   FOR i := 1 TO nEpochs
      nTotalLoss := 0
      
      // Learning rate scheduling: reducir gradualmente
      nCurrentLR := nLearningRate
      IF i > 400
         nCurrentLR := nLearningRate * 0.8
      ENDIF
      IF i > 800
         nCurrentLR := nLearningRate * 0.5
      ENDIF
      
      // Entrenar con cada ejemplo del dataset
      FOR j := 1 TO Len(aTrainingData)
         aInput := aTrainingData[j][1]
         aTarget := aTrainingData[j][2]
         
         // Convertir tokens a embeddings
         mInput := CreateMatrixFromTokens(aInput, mEmbeddings)
         mTarget := CreateMatrixFromTokens(aTarget, mEmbeddings)
         
         // Añadir positional encoding
         mInput := HB_MATRIXADD(mInput, mPositionalEncoding)
         
         // Forward pass con capa de clasificación
         oModel:ZeroGrads()
         mTransformerOutput := oModel:Forward(mInput)
         
         // Aplicar capa de clasificación solo en la posición 7 (respuesta)
         mResponseVector := {mTransformerOutput[7]}  // Solo la posición de respuesta
         mLogits := HB_MATRIXMULTIPLY(mResponseVector, mClassificationWeights)
         mLogits := HB_MATRIXADDBROADCAST(mLogits, mClassificationBias)
         mProbabilities := HB_SOFTMAX(mLogits)
         
         // El target es el índice del token correcto
         nTargetToken := aTarget[7]
         
         IF Empty(mProbabilities)
            ? "ERROR: Forward pass falló en época", i
            QUIT
         ENDIF
         
         // Calcular cross-entropy loss con regularización de entropía
         nLoss := -Log(mProbabilities[1][nTargetToken + 1])  // Cross-entropy básico
         
         // Añadir regularización de entropía para forzar diversidad
         nEntropyRegularization := 0.0
         FOR k := 1 TO nVocabSize
            nProb := mProbabilities[1][k]
            IF nProb > 0.001  // Evitar log(0)
               nEntropyRegularization -= nProb * Log(nProb)
            ENDIF
         NEXT
         // Factor de entropía: penalizar cuando todas las probabilidades son similares
         nLoss := nLoss - 0.05 * nEntropyRegularization  // Reducir factor de entropía
         
         // Backward pass - gradiente de cross-entropy
         mProbGradient := HB_MATRIXCLONE(mProbabilities)
         mProbGradient[1][nTargetToken + 1] -= 1.0  // Gradiente de cross-entropy: p - 1
         
         // Backward a través de la capa de clasificación
         mResponseGradient := HB_MATRIXMULTIPLY(mProbGradient, HB_MATRIXTRANSPOSE(mClassificationWeights))
         
         // Crear gradiente completo para el transformer (ceros excepto posición 7)
         dLoss := HB_MATRIXZERO(nSeqLen, nEmbedDim)
         dLoss[7] := mResponseGradient[1]  // Solo la posición de respuesta tiene gradiente
         
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
         
         // IMPORTANTE: Actualizar los embeddings con el gradiente correcto
         UpdateEmbeddings(mEmbeddings, aInput, mInputGradient, nCurrentLR)
         
         // Actualizar modelo transformer
         oModel:Update(nCurrentLR)
         
         // Actualizar capa de clasificación con momentum
         mWeightGradient := HB_MATRIXMULTIPLY(HB_MATRIXTRANSPOSE(mResponseVector), mProbGradient)
         
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
      
      IF i % 200 == 0 .OR. i <= 10 .OR. i == 600 .OR. i == 1200
         cPhaseDesc := ""
         IF nPhase == 1
            cPhaseDesc := " (FASE GREEN)"
         ELSEIF nPhase == 2
            cPhaseDesc := " (FASE BLUE)"
         ELSE
            cPhaseDesc := " (FASE TODOS)"
         ENDIF
         ? "Época", PadR(Str(i,5),6), "-> Loss promedio:", ;
           Transform(nTotalLoss, "@E 9.999999"), "LR:", Transform(nCurrentLR, "@E 9.9999"), cPhaseDesc
      ENDIF
   NEXT
   
   ? Replicate("-", 70)
   ? ""
   
   // --- VERIFICACIÓN FINAL ---
   ? "========================================================================="
   ? "VERIFICACIÓN: Predicciones del modelo entrenado"
   ? "========================================================================="
   ? ""
   
   FOR j := 1 TO Len(aTrainingData)
      aInput := aTrainingData[j][1]
      aTarget := aTrainingData[j][2]
      
      mInput := CreateMatrixFromTokens(aInput, mEmbeddings)
      mInput := HB_MATRIXADD(mInput, mPositionalEncoding)
      
      mTransformerOutput := oModel:Forward(mInput)
      
      // Aplicar capa de clasificación para obtener probabilidades
      mResponseVector := {mTransformerOutput[7]}
      mLogits := HB_MATRIXMULTIPLY(mResponseVector, mClassificationWeights)
      mLogits := HB_MATRIXADDBROADCAST(mLogits, mClassificationBias)
      mProbabilities := HB_SOFTMAX(mLogits)
      
      // Encontrar el token con mayor probabilidad
      nPredictedToken := 0
      nMaxProb := 0
      FOR k := 1 TO nVocabSize
         IF mProbabilities[1][k] > nMaxProb
            nMaxProb := mProbabilities[1][k]
            nPredictedToken := k - 1
         ENDIF
      NEXT
      
      ? "Pregunta " + Str(j,1) + ":"
      ? "  Input:      ", TokensToWords(aInput, aVocab)
      ? "  Target:     ", aVocab[aTarget[7] + 1]  // Solo la respuesta (posición 7)
      ? "  Predicción: ", aVocab[nPredictedToken + 1], "(prob:", Transform(nMaxProb, "@E 9.99"), ")"
      
      IF aTarget[7] == nPredictedToken
         ? "  ✓ CORRECTO"
      ELSE
         ? "  ✗ INCORRECTO"
      ENDIF
      ? ""
   NEXT
   
   // --- MOSTRAR EVOLUCIÓN DE LOS EMBEDDINGS ---
   ? "========================================================================="
   ? "ANÁLISIS: ¿Qué aprendieron los embeddings?"
   ? "========================================================================="
   ? ""
   ? "Embedding final del token 'SKY' (índice 5):"
   ? "  ", HB_ValToExp(mEmbeddings[6])
   ? ""
   ? "Distancias entre embeddings relacionados:"
   ? "  SKY <-> OCEAN:  ", EuclideanDistance(mEmbeddings[6], mEmbeddings[9])
   ? "  SKY <-> GRASS:  ", EuclideanDistance(mEmbeddings[6], mEmbeddings[7])
   ? "  BLUE <-> GREEN: ", EuclideanDistance(mEmbeddings[10], mEmbeddings[11])
   ? ""
   ? "Los embeddings deberían mostrar que SKY y OCEAN están más cerca"
   ? "(ambos son BLUE) que SKY y GRASS (colores diferentes)."
   ? ""
   
RETURN


/*
* =======================================================================
* FUNCIONES PARA EMBEDDINGS DENSOS
* =======================================================================
*/

/*
* Crea embeddings densos con inicialización Gaussiana N(0, 0.02)
* Esta es la inicialización estándar en transformers modernos
*/
STATIC FUNCTION CreateDenseEmbeddings(nVocabSize, nEmbedDim)
   LOCAL mEmbeddings := {}
   LOCAL i, j, aRow
   LOCAL nStdDev := 0.02  // Inicialización ligeramente más alta para más diversidad
   
   FOR i := 1 TO nVocabSize
      aRow := {}
      FOR j := 1 TO nEmbedDim
         // Inicialización Gaussiana con Box-Muller transform
         AAdd(aRow, GaussianRandom(0, nStdDev))
      NEXT
      AAdd(mEmbeddings, aRow)
   NEXT
   
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
STATIC FUNCTION UpdateEmbeddings(mEmbeddings, aTokens, mInputGradient, nLR)
   LOCAL i, nToken, j
   
   // Para cada token en la secuencia de entrada
   FOR i := 1 TO Len(aTokens)
      nToken := aTokens[i]
      
      IF nToken > 0  // Ignorar padding
         // Actualizar cada dimensión del embedding usando el gradiente que viene del backprop
         FOR j := 1 TO Len(mEmbeddings[nToken + 1])
            IF i <= Len(mInputGradient) .AND. j <= Len(mInputGradient[i])
               // Gradiente estándar con momentum implícito
               mEmbeddings[nToken + 1][j] -= nLR * mInputGradient[i][j]
            ENDIF
         NEXT
      ENDIF
   NEXT
   
RETURN NIL


/*
* Convierte array de tokens en matriz de embeddings densos
*/
STATIC FUNCTION CreateMatrixFromTokens(aTokens, mEmbeddings)
   LOCAL mMatrix := {}
   LOCAL nToken
   
   FOR EACH nToken IN aTokens
      // nToken+1 porque arrays en Harbour son base-1
      AAdd(mMatrix, mEmbeddings[nToken + 1])
   NEXT
   
RETURN mMatrix


/*
* Decodifica output a tokens usando nearest neighbor
* MEJORA: Solo considera posiciones no-padding para la predicción
*/
STATIC FUNCTION DecodeOutputToTokens(mOutput, mEmbeddings)
   LOCAL aTokens := {}
   LOCAL aRow, nBestToken, nMinDist, nToken, nDist
   LOCAL i
   
   FOR i := 1 TO Len(mOutput)
      aRow := mOutput[i]
      nMinDist := 999999
      nBestToken := 0
      
      // Solo buscar entre tokens válidos (no padding)
      FOR nToken := 1 TO Len(mEmbeddings) - 1  // Evitar PAD (token 0)
         nDist := EuclideanDistanceSq(aRow, mEmbeddings[nToken + 1])
         IF nDist < nMinDist
            nMinDist := nDist
            nBestToken := nToken
         ENDIF
      NEXT
      
      AAdd(aTokens, nBestToken)
   NEXT
   
RETURN aTokens


/*
* Distancia Euclidiana al cuadrado (más eficiente)
*/
STATIC FUNCTION EuclideanDistanceSq(aVec1, aVec2)
   LOCAL nSumSq := 0
   LOCAL i, nLen
   
   nLen := Min(Len(aVec1), Len(aVec2))
   
   FOR i := 1 TO nLen
      nSumSq += (aVec1[i] - aVec2[i])^2
   NEXT
   
RETURN nSumSq


/*
* Distancia Euclidiana normal
*/
STATIC FUNCTION EuclideanDistance(aVec1, aVec2)
RETURN Sqrt(EuclideanDistanceSq(aVec1, aVec2))


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
   LOCAL i, j, aRow, nAngle
   
   FOR i := 0 TO nSeqLen - 1
      aRow := {}
      FOR j := 0 TO nEmbedDim - 1
         nAngle := i / ( ( 10000 ^ (2.0 * Int(j/2)) ) / nEmbedDim)
         
         IF j % 2 == 0
            AAdd(aRow, Sin(nAngle))
         ELSE
            AAdd(aRow, Cos(nAngle))
         ENDIF
      NEXT
      AAdd(mPE, aRow)
   NEXT
   
RETURN mPE
