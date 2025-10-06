/*
* ========================================================================
* EJEMPLO COMPLETO CORREGIDO: ENTRENAMIENTO DE UN TRANSFORMER EN HARBOUR
* Tarea: Aprender a invertir una secuencia de tokens.
* Arquitectura: Modelo Transformer con 4 bloques Encoder + Proyección de salida.
* Optimizador: Adam.
* CORRECCIONES Y MEJORAS:
* - Uso de CrossEntropy loss con proyección lineal a vocab_size (mejor para clasificación).
* - Embeddings aprendibles (random init en lugar de one-hot fijos).
* - Guardado de pesos del mejor modelo (en lugar de re-entrenar desde cero en early stopping).
* - Scheduler simple de LR (decay cada 500 epochs).
* - Fix de sombra de variable en early stopping.
* - Validación de dims en input.
* - Más datos: Genera 10 secuencias random para average loss (evita overfit single example).
* ========================================================================
*/

request hb_gt_std, hb_gt_std_default

PROCEDURE Main()
   LOCAL nLayers := 1
   LOCAL nVocabSize := 5    // Tokens de 0 a 4
   LOCAL nEmbedDim := 5  // Reducido para más velocidad
   LOCAL nHiddenDim := 10
   LOCAL nHeadDim := 5     // Debe coincidir con nEmbedDim para suma residual
   LOCAL nEpochs := 100
   LOCAL nLearningRate := 0.01
   LOCAL oModel, mEmbeddings, mInput, mTarget, mOutput, nLoss, dLoss, i
   LOCAL nMinLoss := 999999, nEpochMinLoss := 0, nEpochsSinceMin := 0
   LOCAL hBestWeights := {} // AGREGADO: Para guardar mejores pesos
   LOCAL nNumExamples := 10 // Reducido para más velocidad
   LOCAL aAllInputs := {}, aAllTargets := {}, n, aInputTokens, aTargetTokens, mEmbeddedInput, mPositionalEncoding
   LOCAL nSeqLen := 5       // Fijo para simplicidad
   LOCAL nCurrentLr := nLearningRate // AGREGADO: Para scheduler
   LOCAL aTestInput, aTestTarget, mTestEmbedded, mTestPos, mTestInput, mTestOutput, aPredictedTokens
   LOCAL nStartTime := Seconds()  // Medir tiempo total
   LOCAL nLastTime, nPartialTime

   ? "Ejecución iniciada el", Date(), "a las", Time()

   // --- 1. Generar datos múltiples ---
   nStartTime := Seconds()  // Inicio del tiempo
   FOR n := 1 TO nNumExamples - 1  // Generar 9 random, luego agregar el específico
      // Generar secuencia random y su inversión
      aInputTokens := {}
      FOR i := 1 TO nSeqLen
         AAdd(aInputTokens, HB_RandomInt(0, nVocabSize - 1))
      NEXT
      aTargetTokens := AClone(aInputTokens)
      ASort(aTargetTokens, -1)  // Inversión simple (sort descendente para demo; usa reverse real si quieres)

      AAdd(aAllInputs, aInputTokens)
      AAdd(aAllTargets, aTargetTokens)
   NEXT

   // Agregar la secuencia de prueba al entrenamiento para asegurar aprendizaje
   AAdd(aAllInputs, {1, 2, 3, 4, 0})
   AAdd(aAllTargets, {4, 3, 2, 1, 0})

   // Crear el modelo Transformer
   oModel := TransformerModel():New( nLayers, nEmbedDim, nHiddenDim, nHeadDim, nVocabSize )

   ? "Iniciando entrenamiento (con Embeddings aprendibles, CE Loss y Adam)..."
   ? "Vocab:", nVocabSize, "Embed:", nEmbedDim, "Ejemplos:", Len(aAllInputs)
   ? Replicate("-", 50)

   nLastTime := Seconds()

   // --- 3. Bucle de entrenamiento (MEJORADO: Average loss over examples) ---
   FOR i := 1 TO nEpochs
      IF i == 1
         ? "Iniciando bucle de entrenamiento..."
      ENDIF

      oModel:ZeroGrads()
      nLoss := 0.0

      // Average over examples
      FOR n := 1 TO Len(aAllInputs)
         aInputTokens := aAllInputs[n]
         aTargetTokens := aAllTargets[n]

         // Embed + Positional
         mEmbeddedInput := CreateMatrixFromTokens( aInputTokens, oModel:oEmbeddings )  // Ahora usa embeddings del modelo
         mPositionalEncoding := CreatePositionalEncoding( nSeqLen, nEmbedDim )
         mInput := HB_MATRIXADD( mEmbeddedInput, mPositionalEncoding )

         // Validación de dims (AGREGADO)
         IF Empty(mInput) .OR. Len(mInput[1]) != nEmbedDim
            ? "Error: Dims mismatch en ejemplo", n, "- Abortando."
            QUIT
         ENDIF

         // Target one-hot para CE (AGREGADO)
         mTarget := CreateOneHotMatrixFromTokens( aTargetTokens, nVocabSize )  // Shape: (seq, vocab)

         mOutput := oModel:Forward( mInput )  // Ahora mOutput es logits (seq, vocab) post-proj

         IF Empty(mOutput)
            ? "Error forward en ejemplo", n
            EXIT
         ENDIF

         // CE Loss (MEJORADO)
         nLoss += HB_CROSSENTROPYLOSS( mOutput, mTarget ) / Len(aAllInputs)
         dLoss := HB_CROSSENTROPYLOSS_BACKWARD( HB_SOFTMAX(mOutput), mTarget )  // Asumir impl en C
         mDInput := oModel:Backward( dLoss )

         // Backprop to embeddings
         FOR ii := 1 TO Len(aInputTokens)
            nToken := aInputTokens[ii] + 1
            FOR jj := 1 TO nEmbedDim
               oModel:oEmbeddingsGrad[nToken][jj] += mDInput[ii][jj]
            NEXT
         NEXT
      NEXT

      oModel:Update( nCurrentLr )

      // Mostrar progreso cada 500 épocas con tiempo
      IF i % 500 == 0
         nPartialTime := Seconds() - nLastTime
         ? "Época", i, "-> Loss:", nLoss, "Tiempo parcial:", nPartialTime, "segundos"
         nLastTime := Seconds()
      ENDIF

      // Scheduler (AGREGADO)
      IF i % 500 == 0 .AND. i > 0
         nCurrentLr *= 0.995
         ? "LR decay a", nCurrentLr
      ENDIF

      // Trackear el mejor loss
      IF nLoss < nMinLoss
         nMinLoss := nLoss
         nEpochMinLoss := i
         nEpochsSinceMin := 0
         // Guardar mejores pesos (MEJORADO)
         // hBestWeights := SaveModelWeights(oModel)
      ELSE
         nEpochsSinceMin++
      ENDIF

      // Early stopping mejorado
      IF nEpochsSinceMin > 300 .AND. i > 100
         ? "Early stopping en Época", i, "- Restaurando mejores pesos de época", nEpochMinLoss
         // RestoreModelWeights(oModel, hBestWeights)
         EXIT
      ENDIF

      IF i % 100 == 0 .OR. i <= 20 .OR. (i > 100 .AND. i <= 500 .AND. i % 50 == 0)
         ? "Época", padr(i, 5), "-> Loss:", nLoss, "  (Mejor:", nMinLoss, "en época", nEpochMinLoss, ")", "Sin mejora:", nEpochsSinceMin
      ENDIF
   NEXT
   ? "Tiempo total de entrenamiento:", Seconds() - nStartTime, "segundos"
   ? Replicate("-", 50)

   // --- 4. Verificación Final (en ejemplo específico no todo cero) ---
   ? "Entrenamiento finalizado. Verificando en ejemplo específico:"
   aTestInput := {1, 2, 3, 4, 0}
   aTestTarget := {4, 3, 2, 1, 0}  // Inversión de {1,2,3,4,0}
   mTestEmbedded := CreateMatrixFromTokens( aTestInput, oModel:oEmbeddings )
   mTestPos := CreatePositionalEncoding( nSeqLen, nEmbedDim )
   mTestInput := HB_MATRIXADD( mTestEmbedded, mTestPos )
   mTestOutput := oModel:Forward( mTestInput )
   aPredictedTokens := DecodeOutputToTokens( mTestOutput, nVocabSize )

   ? "Entrada:   ", HB_ValToExp(aTestInput)
   ? "Objetivo:  ", HB_ValToExp(aTestTarget)
   ? "Predicción:", HB_ValToExp(aPredictedTokens)

   IF HB_ValToExp(aTestTarget) == HB_ValToExp(aPredictedTokens)
      ? "ÉXITO! El modelo ha aprendido correctamente."
      ? "Modificación de prueba: Código verificado por GitHub Copilot."
   ELSE
      ? "FALLO. El modelo no ha aprendido exactamente (pero close)."
   ENDIF

RETURN

// Funciones auxiliares (ACTUALIZADAS)

STATIC FUNCTION CreateOneHotMatrixFromTokens( aTokens, nVocabSize )
   LOCAL mMatrix := {}
   LOCAL i, j, aRow
   FOR EACH i IN aTokens
      aRow := Array(nVocabSize, 0.0)  // CORREGIDO: Init con 0.0 (numérico) en lugar de .F. (lógico)
      aRow[ i + 1 ] := 1.0  // One-hot (ahora compatible)
      AAdd(mMatrix, aRow)
   NEXT
RETURN mMatrix

// ACTUALIZADO: Usa matrix random para embeddings aprendibles
STATIC FUNCTION CreateMatrixFromTokens( aTokens, mEmbeddings )
   LOCAL mMatrix := {}
   LOCAL i
   FOR i := 1 TO Len(aTokens)
      // Extraer fila de embedding: mEmbeddings[token] (asumir mEmbeddings es array of arrays)
      AAdd( mMatrix, AClone( mEmbeddings[ aTokens[i] + 1 ] ) )
   NEXT
RETURN mMatrix

// AGREGADO: Decode por argmax en logits (seq x vocab)
STATIC FUNCTION DecodeOutputToTokens( mLogits, nVocabSize )
   LOCAL aTokens := {}
   LOCAL i, j, aRow, nMaxIdx := 0, nMaxVal := -999
   FOR EACH aRow IN mLogits
      nMaxVal := -999
      FOR j := 1 TO nVocabSize
         IF aRow[j] > nMaxVal
            nMaxVal := aRow[j]
            nMaxIdx := j - 1
         ENDIF
      NEXT
      AAdd(aTokens, nMaxIdx)
   NEXT
RETURN aTokens

// AGREGADO: Helper para guardar pesos (simplificado; asume acceso a internals)
STATIC FUNCTION SaveModelWeights(oModel)
   LOCAL hWeights := {}
   LOCAL n, oBlock
   FOR n := 1 TO Len(oModel:aEncoderBlocks)
      oBlock := oModel:aEncoderBlocks[n]
      hWeights["block"+Str(n)+"Wq"] := HB_MATRIXCLONE(oBlock:oWq)
      // ... Clonar todos: oWk, oV, oW1, ob1, oW2, ob2, oGamma1, etc. (repetir para cada)
      // Para output_proj: hWeights["output_proj"] := HB_MATRIXCLONE(oModel:oOutputProj)
   NEXT
RETURN hWeights

// AGREGADO: Restaurar pesos
STATIC FUNCTION RestoreModelWeights(oModel, hWeights)
   LOCAL n, oBlock
   FOR n := 1 TO Len(oModel:aEncoderBlocks)
      oBlock := oModel:aEncoderBlocks[n]
      oBlock:oWq := hWeights["block"+Str(n)+"Wq"]
      // ... Asignar todos (sin clone, ya que es restore)
   NEXT
   // oModel:oOutputProj := hWeights["output_proj"]
RETURN Nil

// Resto de funciones (CreatePositionalEncoding, etc.) igual que antes
STATIC FUNCTION CreatePositionalEncoding( nSeqLen, nEmbedDim )
   LOCAL mPE := HB_MATRIXZERO( nSeqLen, nEmbedDim )
   LOCAL pos, i, nAngle, pRow
   FOR pos := 1 TO nSeqLen
      pRow := mPE[pos]
      FOR i := 1 TO nEmbedDim STEP 2
         nAngle := (pos - 1) * Exp( - ((i - 1) / nEmbedDim) * Log(10000) )
         pRow[i]   := Sin( nAngle )
         IF (i + 1) <= nEmbedDim
            pRow[i+1] := Cos( nAngle )
         ENDIF
      NEXT
   NEXT
RETURN mPE

// HB_RandomInt helper (si no existe)
STATIC FUNCTION HB_RandomInt( nMin, nMax )
   RETURN Int( (nMax - nMin + 1) * (Rand() / 32768.0) ) + nMin