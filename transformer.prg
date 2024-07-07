#include "hbclass.ch"

#include "hbclass.ch"

CLASS Transformer
   PROTECTED:
      DATA nHeads
      DATA nModelDim
      DATA nFeedForwardDim
      DATA nMaxSeqLength
      DATA aPositionalEncoding
      DATA aWeights
      DATA aBiases
      DATA aGamma
      DATA aBeta
      
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
      METHOD LinearProjection(aInput, nOutputDim, cType)
      METHOD MatMul(aMatrix1, aMatrix2)
      METHOD Transpose(aMatrix)
      METHOD SoftMax(aVector)
      METHOD ReLU(aInput)
      METHOD Mean(aVector)
      METHOD Variance(aVector, nMean)
      METHOD ElementWiseAddition(aMatrix1, aMatrix2)
      METHOD ElementWiseMultiplication(aMatrix1, aMatrix2)
      METHOD GeneratePositionalEncoding()
      METHOD InitializeParameters()
      METHOD GenerateWeights(nInputDim, nOutputDim, cType)

ENDCLASS

METHOD New(nHeads, nModelDim, nFeedForwardDim, nMaxSeqLength) CLASS Transformer
   ::nHeads := nHeads
   ::nModelDim := nModelDim
   ::nFeedForwardDim := nFeedForwardDim
   ::nMaxSeqLength := nMaxSeqLength
   ::GeneratePositionalEncoding()
   ::InitializeParameters()
RETURN Self

METHOD InitializeParameters() CLASS Transformer
   LOCAL i
   ::aWeights := {}
   ::aBiases := {}

   // Weights and biases for Q, K, V projections
   FOR i := 1 TO ::nHeads
      AAdd(::aWeights, ::GenerateWeights(::nModelDim, ::nModelDim / ::nHeads, "attention"))
      AAdd(::aBiases, Array(::nModelDim / ::nHeads))
   NEXT

   // Weights and bias for output projection
   AAdd(::aWeights, ::GenerateWeights(::nModelDim, ::nModelDim, "output"))
   AAdd(::aBiases, Array(::nModelDim))

   // Weights and biases for feed-forward layers
   AAdd(::aWeights, ::GenerateWeights(::nModelDim, ::nFeedForwardDim, "feedforward1"))
   AAdd(::aBiases, Array(::nFeedForwardDim))
   AAdd(::aWeights, ::GenerateWeights(::nFeedForwardDim, ::nModelDim, "feedforward2"))
   AAdd(::aBiases, Array(::nModelDim))

   // Initialize gamma and beta for layer normalization
   ::aGamma := Array(::nModelDim)
   ::aBeta := Array(::nModelDim)
   AEval(::aGamma, {|x| x := 1})
   AEval(::aBeta, {|x| x := 0})
RETURN NIL

METHOD GenerateWeights(nInputDim, nOutputDim, cType) CLASS Transformer
   LOCAL aWeights := Array(nInputDim)
   LOCAL i, j, nStdDev

   // Xavier/Glorot initialization
   nStdDev := Sqrt(2 / (nInputDim + nOutputDim))

   FOR i := 1 TO nInputDim
      aWeights[i] := Array(nOutputDim)
      FOR j := 1 TO nOutputDim
         aWeights[i][j] := (HB_RANDOM() * 2 - 1) * nStdDev
      NEXT
   NEXT

RETURN aWeights

METHOD Forward(aInput) CLASS Transformer
   LOCAL aEncoded
   LOCAL i
   
   aEncoded := aInput
   FOR i := 1 TO 6  // Example with 6 layers
      aEncoded := ::Encode(aEncoded)
   NEXT
RETURN aEncoded

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
   LOCAL i, j, nSeqLen := Len(aInput)
   
   FOR i := 1 TO nSeqLen
      FOR j := 1 TO ::nModelDim
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

   FOR i := 1 TO ::nHeads
      aQ := ::LinearProjection(aQuery, nHeadDim, "attention_q" + AllTrim(Str(i)))
      aK := ::LinearProjection(aKey, nHeadDim, "attention_k" + AllTrim(Str(i)))
      aV := ::LinearProjection(aValue, nHeadDim, "attention_v" + AllTrim(Str(i)))
      
      aHeadOutput := ::DotProductAttention(aQ, aK, aV)
      AAdd(aHeadOutputs, aHeadOutput)
   NEXT

   aConcatenated := ::ConcatenateHeads(aHeadOutputs)
   aProjected := ::LinearProjection(aConcatenated, ::nModelDim, "output")

RETURN aProjected

METHOD ConcatenateHeads(aHeadOutputs) CLASS Transformer
   LOCAL aConcatenated := {}
   LOCAL i, j, k, nHeadDim := ::nModelDim / ::nHeads

   FOR i := 1 TO Len(aHeadOutputs[1])
      AAdd(aConcatenated, Array(::nModelDim))
      FOR j := 1 TO Len(aHeadOutputs)
         FOR k := 1 TO nHeadDim
            aConcatenated[i][(j - 1) * nHeadDim + k] := aHeadOutputs[j][i][k]
         NEXT
      NEXT
   NEXT

RETURN aConcatenated

METHOD DotProductAttention(aQuery, aKey, aValue) CLASS Transformer
   LOCAL aScores, aSoftMaxScores, aOutput
   LOCAL nDimK := Len(aKey[1])
   LOCAL i, j

   aScores := ::MatMul(aQuery, ::Transpose(aKey))
   
   FOR i := 1 TO Len(aScores)
      FOR j := 1 TO Len(aScores[i])
         aScores[i][j] := aScores[i][j] / Sqrt(nDimK)
      NEXT
   NEXT

   aSoftMaxScores := {}
   AEval(aScores, {|a| AAdd(aSoftMaxScores, ::SoftMax(a))})

   aOutput := ::MatMul(aSoftMaxScores, aValue)

RETURN aOutput

METHOD LinearProjection(aInput, nOutputDim, cType) CLASS Transformer
   LOCAL aOutput := {}
   LOCAL i, j, nWeightIndex, nBiasIndex

   // Get predefined weights and biases
   nWeightIndex := AScan(::aWeights, {|a| Len(a[1]) == nOutputDim})
   nBiasIndex := AScan(::aBiases, {|a| Len(a) == nOutputDim})

   IF nWeightIndex == 0 .OR. nBiasIndex == 0
      // Handle error: weights or biases not found
      RETURN NIL
   ENDIF

   FOR i := 1 TO Len(aInput)
      AAdd(aOutput, Array(nOutputDim))
      FOR j := 1 TO nOutputDim
         aOutput[i][j] := ::MatMul({aInput[i]}, ::aWeights[nWeightIndex])[1][j] + ::aBiases[nBiasIndex][j]
      NEXT
   NEXT

RETURN aOutput

METHOD LayerNorm(aInput) CLASS Transformer
   LOCAL nMean, nVar, aNorm
   LOCAL i, j

   aNorm := Array(Len(aInput))
   FOR i := 1 TO Len(aInput)
      nMean := ::Mean(aInput[i])
      nVar := ::Variance(aInput[i], nMean)
      aNorm[i] := Array(Len(aInput[i]))
      FOR j := 1 TO Len(aInput[i])
         aNorm[i][j] := ((aInput[i][j] - nMean) / Sqrt(nVar + 1e-6)) * ::aGamma[j] + ::aBeta[j]
      NEXT
   NEXT

RETURN aNorm

METHOD FeedForward(aInput) CLASS Transformer
   LOCAL aHidden, aOutput
   
   aHidden := ::ReLU(::LinearProjection(aInput, ::nFeedForwardDim, "feedforward1"))
   aOutput := ::LinearProjection(aHidden, ::nModelDim, "feedforward2")

RETURN aOutput

METHOD SoftMax(aVector) CLASS Transformer
   LOCAL nSum := 0
   LOCAL aOutput := Array(Len(aVector))
   LOCAL nMax, i
   
   // Find max for numerical stability
   nMax := aVector[1]
   FOR i := 2 TO Len(aVector)
      IF aVector[i] > nMax
         nMax := aVector[i]
      ENDIF
   NEXT
   
   // Calculate exponential and sum
   FOR i := 1 TO Len(aVector)
      aOutput[i] := Exp(aVector[i] - nMax)
      nSum += aOutput[i]
   NEXT
   
   // Normalize
   AEval(aOutput, {|x, i| aOutput[i] := x / nSum})

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

