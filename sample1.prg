/*
* ========================================================================
* EJEMPLO COMPLETO: ENTRENAMIENTO DE UN TRANSFORMER EN HARBOUR
* Tarea: Aprender a invertir una secuencia de tokens.
* Arquitectura: Modelo Transformer con 4 bloques Encoder.
* Optimizador: Adam.
* ========================================================================
*/

PROCEDURE Main()
   LOCAL nLayers := 2       // Reducir complejidad: menos capas
   LOCAL nVocabSize := 5    // Tokens de 0 a 4
   LOCAL nEmbedDim := 16    // Dimensión de los vectores
   LOCAL nHiddenDim := 32   // Reducir dimensión oculta
   LOCAL nHeadDim := nEmbedDim
   LOCAL nEpochs := 2000    // Más épocas para asegurar la convergencia
   LOCAL nLearningRate := 0.005 // Learning rate moderado
   LOCAL oModel, mEmbeddedInput, mPositionalEncoding, mInput, mTarget, mOutput, nLoss, dLoss, i
   LOCAL nMinLoss := 999999, nEpochMinLoss := 0, nEpochsSinceMin := 0
   LOCAL oBestModel := NIL

   // --- 1. Definir el problema y los datos ---
   LOCAL aInputTokens  := { 1, 2, 3, 4, 0 }
   LOCAL aTargetTokens := { 4, 3, 2, 1, 0 }
   LOCAL nSeqLen := Len(aInputTokens)

   LOCAL mEmbeddings := CreateOneHotEmbeddings( nVocabSize, nEmbedDim ), aPredictedTokens
   
   mEmbeddedInput  := CreateMatrixFromTokens( aInputTokens, mEmbeddings )
   mTarget := CreateMatrixFromTokens( aTargetTokens, mEmbeddings )

   // Crear y añadir la Codificación Posicional
   mPositionalEncoding := CreatePositionalEncoding( nSeqLen, nEmbedDim )
   mInput := HB_MATRIXADD( mEmbeddedInput, mPositionalEncoding )

   // --- 2. Crear el modelo ---
   oModel := TransformerModel():New( nLayers, nEmbedDim, nHiddenDim, nHeadDim )

   ? "Iniciando entrenamiento (con Positional Encoding y Adam)..."
   ? "Entrada:", HB_ValToExp(aInputTokens)
   ? "Objetivo:", HB_ValToExp(aTargetTokens)
   ? "Configuración: Épocas =", nEpochs, ", LR =", nLearningRate
   ? Replicate("-", 50)

   // --- 3. Bucle de entrenamiento ---
   FOR i := 1 TO nEpochs
      IF i == 1
         ? "Iniciando bucle de entrenamiento..."
      ENDIF
      
      oModel:ZeroGrads()
      mOutput := oModel:Forward( mInput )

      // ====> ?VERIFICACIÓN CRÍTICA EN HARBOUR! <====
      // Si el forward pass falló, mOutput estará vacío.
      IF Empty(mOutput)
         ? "---------------------------------------------------------"
         ? "?ERROR CRÍTICO! El forward pass del modelo ha fallado."
         ? "Causa más probable: Discordancia de dimensiones en las"
         ? "matrices dentro de la arquitectura del modelo."
         ? "Revisa nEmbedDim, nHiddenDim y nHeadDim."
         ? "---------------------------------------------------------"
         QUIT // Terminar el programa
      ENDIF

      nLoss   := HB_MSE_LOSS( mOutput, mTarget )
      dLoss   := HB_MSE_LOSS_BACKWARD( mOutput, mTarget )
      oModel:Backward( dLoss )
      oModel:Update( nLearningRate )

      // Trackear el mejor loss
      IF nLoss < nMinLoss
         nMinLoss := nLoss
         nEpochMinLoss := i
         nEpochsSinceMin := 0
      ELSE
         nEpochsSinceMin++
      ENDIF
      
      // Early stopping: si no mejora en 500 épocas, detener
      IF nEpochsSinceMin > 200 .AND. i > 100
         ? "Early stopping en época", i, "- No hay mejora desde época", nEpochMinLoss
         ? "Reentrenando hasta el mejor punto..."
         // Recrear y reentrenar hasta el mejor epoch
         oModel := TransformerModel():New( nLayers, nEmbedDim, nHiddenDim, nHeadDim )
         FOR i := 1 TO nEpochMinLoss
            oModel:ZeroGrads()
            mOutput := oModel:Forward( mInput )
            IF Empty(mOutput)
               EXIT
            ENDIF
            nLoss := HB_MSE_LOSS( mOutput, mTarget )
            dLoss := HB_MSE_LOSS_BACKWARD( mOutput, mTarget )
            oModel:Backward( dLoss )
            oModel:Update( nLearningRate )
         NEXT
         ? "Modelo restaurado al mejor punto (época", nEpochMinLoss, ")"
         EXIT
      ENDIF

      IF i % 100 == 0 .OR. i <= 10
         ? "Época", padr(i, 5), "-> Loss:", nLoss, "  (Mejor:", nMinLoss, "en época", nEpochMinLoss, ")", "SinMejora:", nEpochsSinceMin
      ENDIF
   NEXT
   ? Replicate("-", 50)

   // --- 4. Verificación Final ---
   ? "Entrenamiento finalizado. Verificando resultado:"
   mOutput := oModel:Forward( mInput )
   aPredictedTokens := DecodeOutputToTokens( mOutput, mEmbeddings )

   ? "Entrada:   ", HB_ValToExp(aInputTokens)
   ? "Objetivo:  ", HB_ValToExp(aTargetTokens)
   ? "Predicción:", HB_ValToExp(aPredictedTokens)
   
   // Mostrar las distancias del último token para debug
   ? "Distancias del último token a cada embedding:"
   FOR i := 0 TO nVocabSize - 1
      ? "  Token", i, ":", EuclideanDistSq(mOutput[5], mEmbeddings[i+1])
   NEXT
   ?
   IF HB_ValToExp(aTargetTokens) == HB_ValToExp(aPredictedTokens)
      ? "?ÉXITO! El modelo ha aprendido a invertir la secuencia."
   ELSE
      ? "FALLO. El modelo no ha aprendido la tarea correctamente."
   ENDIF

RETURN

/*
* Crea una matriz de embedding con codificación one-hot.
*/
STATIC FUNCTION CreateOneHotEmbeddings( nVocabSize, nEmbedDim )
   LOCAL mEmbeddings := {}
   LOCAL i, j, aRow
   // Asegurarse de que la dimensión del embedding sea suficiente
   IF nEmbedDim < nVocabSize
      nEmbedDim := nVocabSize
   ENDIF
   FOR i := 0 TO nVocabSize - 1
      aRow := {}
      FOR j := 1 TO nEmbedDim
         AAdd(aRow, 0)
      NEXT
      aRow[ i + 1 ] := 1
      AAdd( mEmbeddings, aRow )
   NEXT
RETURN mEmbeddings

/*
* Convierte un array de IDs de token en una matriz de vectores.
*/
STATIC FUNCTION CreateMatrixFromTokens( aTokens, mEmbeddings )
   LOCAL mMatrix := {}
   AEval( aTokens, {|nToken| AAdd( mMatrix, mEmbeddings[ nToken + 1 ] ) } )
RETURN mMatrix

/*
* Decodifica la matriz de salida para obtener los IDs de token (argmax).
*/
STATIC FUNCTION DecodeOutputToTokens( mOutputMatrix, mEmbeddings )
   LOCAL aTokens := {}
   LOCAL aRow, nBestToken, nMinDist, nToken, nDist

   FOR EACH aRow IN mOutputMatrix
      nMinDist := 999999
      nBestToken := -1
      
      FOR nToken := 0 TO Len(mEmbeddings) - 1
         nDist := EuclideanDistSq( aRow, mEmbeddings[nToken+1] )
         IF nDist < nMinDist
            nMinDist := nDist
            nBestToken := nToken
         ENDIF
      NEXT
      AAdd(aTokens, nBestToken)
   NEXT
RETURN aTokens

STATIC FUNCTION EuclideanDistSq( aVec1, aVec2 )
   LOCAL nSumSq := 0, i, nLen
   
   nLen := Min(Len(aVec1), Len(aVec2))
   
   FOR i := 1 TO nLen
      nSumSq += (aVec1[i] - aVec2[i])^2
   NEXT
   
   // Penalizar si las longitudes son diferentes
   IF Len(aVec1) != Len(aVec2)
      nSumSq += Abs(Len(aVec1) - Len(aVec2)) * 1000
   ENDIF
RETURN nSumSq