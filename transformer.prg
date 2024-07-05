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
      METHOD MatMul(aMatrix1, aMatrix2)
      METHOD Transpose(aMatrix)  
      METHOD SoftMax(aVector)
      METHOD FeedForward(aInput)
      METHOD LayerNorm(aInput)
      
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
   // Implementación simplificada de normalización de capa
   // En una implementación real, esto normalizaría los valores de entrada
   RETURN aInput // Placeholder
