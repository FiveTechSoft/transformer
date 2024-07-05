#include "hbclass.ch"

CLASS Transformer
   PROTECTED:
      VAR nHeads
      VAR nModelDim
      VAR nFeedForwardDim
      
   EXPORTED:
      METHOD New(nHeads, nModelDim, nFeedForwardDim)
      METHOD Forward(aInput)
      METHOD MultiHeadAttention(aQuery, aKey, aValue)
      METHOD DotProductAttention(aQuery, aKey, aValue)
      METHOD LinearProjection(aInput, nOutputDim)
      METHOD LinearTransformation(aInput, nOutputDim)
      METHOD MatMul(aMatrix1, aMatrix2)
      METHOD Transpose(aMatrix)  
      METHOD SoftMax(aVector)
      METHOD FeedForward(aInput)
      METHOD LayerNorm(aInput)
      METHOD ReLU(aInput)
      
ENDCLASS

METHOD New(nHeads, nModelDim, nFeedForwardDim) CLASS Transformer
   ::nHeads := nHeads
   ::nModelDim := nModelDim
   ::nFeedForwardDim := nFeedForwardDim
RETURN Self

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
