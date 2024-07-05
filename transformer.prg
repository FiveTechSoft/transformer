#include "hbclass.ch"

CLASS Transformer
   PROTECTED:
      VAR nHeads
      VAR nModelDim
      VAR nFeedForwardDim
      VAR nMaxSeqLength
      VAR aPositionalEncoding
      
   EXPORTED:
      METHOD New(nHeads, nModelDim, nFeedForwardDim, nMaxSeqLength)
      METHOD Forward(aInput)
      METHOD MultiHeadAttention(aQuery, aKey, aValue)
      METHOD FeedForward(aInput)
      METHOD LayerNorm(aInput)
      METHOD PositionalEncoding(aInput)
      METHOD Encode(aInput)
      METHOD Decode(aInput, aEncoderOutput)
      
   PROTECTED:
      METHOD DotProductAttention(aQuery, aKey, aValue)
      METHOD LinearProjection(aInput, nOutputDim)
      METHOD MatMul(aMatrix1, aMatrix2)
      METHOD Transpose(aMatrix)
      METHOD SoftMax(aVector)
      METHOD ReLU(aInput)
      METHOD Mean(aVector)
      METHOD Variance(aVector, nMean)
      METHOD ElementWiseAddition(aMatrix1, aMatrix2)
      METHOD ElementWiseMultiplication(aMatrix1, aMatrix2)
      METHOD GeneratePositionalEncoding()

ENDCLASS

METHOD New(nHeads, nModelDim, nFeedForwardDim, nMaxSeqLength) CLASS Transformer
   ::nHeads := nHeads
   ::nModelDim := nModelDim
   ::nFeedForwardDim := nFeedForwardDim
   ::nMaxSeqLength := nMaxSeqLength
   ::GeneratePositionalEncoding()
RETURN Self

METHOD Forward(aInput) CLASS Transformer
   LOCAL aEncodedInput, aOutput
   
   aEncodedInput := ::Encode(aInput)
   aOutput := ::Decode(aInput, aEncodedInput)
RETURN aOutput

METHOD Encode(aInput) CLASS Transformer
   LOCAL aPositionalInput, aAttOutput, aFeedForwardOutput
   
   aPositionalInput := ::PositionalEncoding(aInput)
   aAttOutput := ::MultiHeadAttention(aPositionalInput, aPositionalInput, aPositionalInput)
   aAttOutput := ::LayerNorm(::ElementWiseAddition(aPositionalInput, aAttOutput))
   
   aFeedForwardOutput := ::FeedForward(aAttOutput)
   aFeedForwardOutput := ::LayerNorm(::ElementWiseAddition(aAttOutput, aFeedForwardOutput))
RETURN aFeedForwardOutput

METHOD Decode(aInput, aEncoderOutput) CLASS Transformer
   LOCAL aPositionalInput, aSelfAttOutput, aCrossAttOutput, aFeedForwardOutput
   
   aPositionalInput := ::PositionalEncoding(aInput)
   aSelfAttOutput := ::MultiHeadAttention(aPositionalInput, aPositionalInput, aPositionalInput)
   aSelfAttOutput := ::LayerNorm(::ElementWiseAddition(aPositionalInput, aSelfAttOutput))
   
   aCrossAttOutput := ::MultiHeadAttention(aSelfAttOutput, aEncoderOutput, aEncoderOutput)
   aCrossAttOutput := ::LayerNorm(::ElementWiseAddition(aSelfAttOutput, aCrossAttOutput))
   
   aFeedForwardOutput := ::FeedForward(aCrossAttOutput)
   aFeedForwardOutput := ::LayerNorm(::ElementWiseAddition(aCrossAttOutput, aFeedForwardOutput))
RETURN aFeedForwardOutput

METHOD PositionalEncoding(aInput) CLASS Transformer
   LOCAL aOutput := AClone(aInput)
   LOCAL i, j
   
   FOR i := 1 TO Len(aInput)
      FOR j := 1 TO Len(aInput[i])
         aOutput[i][j] += ::aPositionalEncoding[i][j]
      NEXT
   NEXT
RETURN aOutput

METHOD GeneratePositionalEncoding() CLASS Transformer
   LOCAL i, j, nPos, nDim
   
   ::aPositionalEncoding := Array(::nMaxSeqLength)
   FOR i := 1 TO ::nMaxSeqLength
      ::aPositionalEncoding[i] := Array(::nModelDim)
      FOR j := 1 TO ::nModelDim
         nPos := i
         nDim := j
         IF nDim % 2 == 0
            ::aPositionalEncoding[i][j] := Cos(nPos / (10000 ^ (nDim / ::nModelDim)))
         ELSE
            ::aPositionalEncoding[i][j] := Sin(nPos / (10000 ^ ((nDim - 1) / ::nModelDim)))
         ENDIF
      NEXT
   NEXT
RETURN NIL

METHOD Forward(aInput) CLASS Transformer
   LOCAL aAttOutput, aFeedForwardOutput
   
   aAttOutput := ::MultiHeadAttention(aInput, aInput, aInput)
   aAttOutput := ::LayerNorm(aInput + aAttOutput)
   
   aFeedForwardOutput := ::FeedForward(aAttOutput)
   aFeedForwardOutput := ::LayerNorm(aAttOutput + aFeedForwardOutput)
RETURN aFeedForwardOutput

METHOD MultiHeadAttention(aQuery, aKey, aValue) CLASS Transformer
   LOCAL nHeadDim := ::nModelDim / ::nHeads
   LOCAL aHeadOutputs := {}
   LOCAL i, aQ, aK, aV, aHeadOutput, aConcatenated, aProjected

   // Dividir en cabezas
   FOR i := 1 TO ::nHeads
      aQ := ::LinearProjection(aQuery, nHeadDim)
      aK := ::LinearProjection(aKey, nHeadDim)
      aV := ::LinearProjection(aValue, nHeadDim)
      
      // Calcular la atención para esta cabeza
      aHeadOutput := ::DotProductAttention(aQ, aK, aV)
      AAdd(aHeadOutputs, aHeadOutput)
   NEXT

   // Concatenar las salidas de todas las cabezas
   aConcatenated := {}
   FOR i := 1 TO Len(aHeadOutputs[1])
      AAdd(aConcatenated, ASize(Array(0), ::nModelDim))
      AFill(aConcatenated[i], 0)
      AEval(aHeadOutputs, {|a, j| AEval(a[i], {|x, k| aConcatenated[i][k + (j-1)*nHeadDim] += x })})
   NEXT

   // Proyección lineal final
   aProjected := ::LinearProjection(aConcatenated, ::nModelDim)

RETURN aProjected

METHOD DotProductAttention(aQuery, aKey, aValue) CLASS Transformer
   LOCAL aScores, aSoftMaxScores, aOutput
   LOCAL nDimK := Len(aKey[1])

   // Calcular puntuaciones de atención
   aScores := ::MatMul(aQuery, ::Transpose(aKey))
   AEval(aScores, {|a| AEval(a, {|x, i| a[i] := x / Sqrt(nDimK) })})

   // Aplicar softmax
   aSoftMaxScores := {}
   AEval(aScores, {|a| AAdd(aSoftMaxScores, ::SoftMax(a))})

   // Calcular el output final
   aOutput := ::MatMul(aSoftMaxScores, aValue)

RETURN aOutput

METHOD LinearProjection(aInput, nOutputDim) CLASS Transformer
   LOCAL aOutput := {}
   LOCAL aWeights := {}
   LOCAL i, j

   // Inicializar pesos aleatorios (en una implementación real, estos serían parámetros entrenables)
   FOR i := 1 TO Len(aInput[1])
      AAdd(aWeights, Array(nOutputDim))
      AEval(aWeights[i], {|x| HB_RANDOM() / HB_RANDOM(1) })
   NEXT

   // Realizar la proyección lineal
   aOutput := ::MatMul(aInput, aWeights)

RETURN aOutput

METHOD LinearTransformation(aInput, nOutputDim) CLASS Transformer
   LOCAL aOutput := {}
   LOCAL aWeights := {}
   LOCAL aBiases := {}
   LOCAL i, j

   // Inicializar pesos aleatorios (en una implementación real, estos serían parámetros entrenables)
   FOR i := 1 TO Len(aInput[1])
      AAdd(aWeights, Array(nOutputDim))
      AEval(aWeights[i], {|x| HB_RANDOM() / HB_RANDOM(1) - 0.5 })
   NEXT

   // Inicializar biases
   aBiases := Array(nOutputDim)
   AEval(aBiases, {|x| HB_RANDOM() / HB_RANDOM(1) - 0.5 })

   // Realizar la transformación lineal: output = input * weights + biases
   aOutput := ::MatMul(aInput, aWeights)

   // Añadir biases
   FOR i := 1 TO Len(aOutput)
      FOR j := 1 TO Len(aOutput[i])
         aOutput[i][j] += aBiases[j]
      NEXT
   NEXT

RETURN aOutput

METHOD MatMul(aMatrix1, aMatrix2) CLASS Transformer
   LOCAL aResult := {}
   LOCAL i, j, k, nSum

   FOR i := 1 TO Len(aMatrix1)
      AAdd(aResult, Array(Len(aMatrix2[1])))
      FOR j := 1 TO Len(aMatrix2[1])
         nSum := 0
         FOR k := 1 TO Len(aMatrix1[1])
            nSum += aMatrix1[i][k] * aMatrix2[k][j]
         NEXT
         aResult[i][j] := nSum
      NEXT
   NEXT

RETURN aResult

METHOD Transpose(aMatrix) CLASS Transformer
   LOCAL aResult := {}
   LOCAL i, j

   FOR i := 1 TO Len(aMatrix[1])
      AAdd(aResult, Array(Len(aMatrix)))
      FOR j := 1 TO Len(aMatrix)
         aResult[i][j] := aMatrix[j][i]
      NEXT
   NEXT

RETURN aResult

METHOD SoftMax(aVector) CLASS Transformer
   LOCAL aResult := AClone(aVector)
   LOCAL nSum := 0
   LOCAL i

   // Aplicar exponencial y sumar
   AEval(aResult, {|x, i| aResult[i] := Exp(x), nSum += aResult[i] })

   // Normalizar
   AEval(aResult, {|x, i| aResult[i] := x / nSum })

RETURN aResult

METHOD FeedForward(aInput) CLASS Transformer
   LOCAL aHidden, aOutput

   // Primera transformación lineal
   aHidden := ::LinearTransformation(aInput, ::nFeedForwardDim)

   // Aplicar ReLU
   aHidden := ::ReLU(aHidden)

   // Segunda transformación lineal
   aOutput := ::LinearTransformation(aHidden, ::nModelDim)

RETURN aOutput

METHOD LayerNorm(aInput) CLASS Transformer
   LOCAL aOutput := {}
   LOCAL aGamma := {}
   LOCAL aBeta := {}
   LOCAL nMean, nVariance, nEpsilon := 1e-8
   LOCAL i, j, aTemp

   // Inicializar parámetros aprendibles gamma y beta
   FOR i := 1 TO Len(aInput[1])
      AAdd(aGamma, 1)  // Inicializado a 1
      AAdd(aBeta, 0)   // Inicializado a 0
   NEXT

   // Normalizar cada vector de entrada
   FOR i := 1 TO Len(aInput)
      // Calcular media y varianza
      nMean := ::Mean(aInput[i])
      nVariance := ::Variance(aInput[i], nMean)

      // Normalizar
      aTemp := Array(Len(aInput[i]))
      FOR j := 1 TO Len(aInput[i])
         aTemp[j] := (aInput[i][j] - nMean) / Sqrt(nVariance + nEpsilon)
      NEXT

      // Aplicar gamma y beta
      FOR j := 1 TO Len(aTemp)
         aTemp[j] := aGamma[j] * aTemp[j] + aBeta[j]
      NEXT

      AAdd(aOutput, aTemp)
   NEXT

RETURN aOutput

METHOD ReLU(aInput) CLASS Transformer
   LOCAL aOutput := AClone(aInput)
   LOCAL i, j

   FOR i := 1 TO Len(aOutput)
      FOR j := 1 TO Len(aOutput[i])
         aOutput[i][j] := Max(0, aOutput[i][j])
      NEXT
   NEXT

RETURN aOutput

METHOD Mean(aVector) CLASS Transformer
   LOCAL nSum := 0
   AEval(aVector, {|x| nSum += x})
RETURN nSum / Len(aVector)

METHOD Variance(aVector, nMean) CLASS Transformer
   LOCAL nSum := 0
   AEval(aVector, {|x| nSum += (x - nMean) ^ 2})
RETURN nSum / Len(aVector)

METHOD ElementWiseAddition(aMatrix1, aMatrix2) CLASS Transformer
   LOCAL aResult := {}
   LOCAL i, j

   FOR i := 1 TO Len(aMatrix1)
      AAdd(aResult, Array(Len(aMatrix1[i])))
      FOR j := 1 TO Len(aMatrix1[i])
         aResult[i][j] := aMatrix1[i][j] + aMatrix2[i][j]
      NEXT
   NEXT

RETURN aResult

METHOD ElementWiseMultiplication(aMatrix1, aMatrix2) CLASS Transformer
   LOCAL aResult := {}
   LOCAL i, j

   FOR i := 1 TO Len(aMatrix1)
      AAdd(aResult, Array(Len(aMatrix1[i])))
      FOR j := 1 TO Len(aMatrix1[i])
         aResult[i][j] := aMatrix1[i][j] * aMatrix2[i][j]
      NEXT
   NEXT

RETURN aResult

