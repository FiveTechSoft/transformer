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
      DATA aGradients
      DATA aLastFeedForwardInput
      DATA aLastAttentionInput
   
   EXPORTED:
      METHOD New(nHeads, nModelDim, nFeedForwardDim, nMaxSeqLength)
      METHOD Forward(aInput)
      METHOD MultiHeadAttention(aQuery, aKey, aValue)
      METHOD FeedForward(aInput)
      METHOD LayerNorm(aInput)
      METHOD PositionalEncoding(aInput)
      METHOD Encode(aInput)
      METHOD Decode(aInput, aEncoderOutput)
      METHOD Backward(aOutputGradient, nLearningRate)
      METHOD UpdateParameters(nLearningRate)
   
   PROTECTED:
      METHOD ConcatenateHeads(aHeadOutputs)
      METHOD DotProductAttention(aQuery, aKey, aValue)
      METHOD LinearProjection(aInput, nOutputDim, cType)
      METHOD LinearTransformation(aInput, nOutputDim)
      METHOD MatAdd( aMatrix1, aMatrix2 )
      METHOD MatMul(aMatrix1, aMatrix2)
      METHOD Transpose(aMatrix)
      METHOD SoftMax(aVector)
      METHOD SplitHeads(aInput)
      METHOD SumGradients(aGradients)
      METHOD ReLU(aInput)
      METHOD Mean(aVector)
      METHOD Variance(aVector, nMean)
      METHOD ElementWiseAddition( aMatrix1, aMatrix2 )
      METHOD ElementWiseMultiplication( aMatrix1, aMatrix2 )
      METHOD GeneratePositionalEncoding()
      METHOD InitializeParameters()
      METHOD GenerateWeights( nInputDim, nOutputDim, cType )
      METHOD BackwardEncode( aGradient )
      METHOD BackwardMultiHeadAttention( aInputGradient, aQuery, aKey, aValue )
      METHOD BackwardFeedForward( aInputGradient, aInput )
      METHOD BackwardLayerNorm( aInputGradient, aInput )
      METHOD BackwardLinearProjection( aInputGradient, aInput, cType )
      METHOD BackwardReLU( aInputGradient, aInput )
      METHOD BackwardSoftMax( aInputGradient, aInput )
   
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
      AAdd(::aBiases, GenerateRandomVector( ::nModelDim / ::nHeads ) )
   NEXT

   // Weights and bias for output projection
   AAdd(::aWeights, ::GenerateWeights(::nModelDim, ::nModelDim, "output"))
   AAdd(::aBiases, GenerateRandomVector(::nModelDim))

   // Weights and biases for feed-forward layers
   AAdd(::aWeights, ::GenerateWeights(::nModelDim, ::nFeedForwardDim, "feedforward1"))
   AAdd(::aBiases, GenerateRandomVector(::nFeedForwardDim))
   AAdd(::aWeights, ::GenerateWeights(::nFeedForwardDim, ::nModelDim, "feedforward2"))
   AAdd(::aBiases, GenerateRandomVector(::nModelDim))

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

METHOD Backward(aOutputGradient, nLearningRate) CLASS Transformer
   LOCAL aGradient, i
   
   ::aGradients := {}
   
   // Assuming we have 6 encoder layers
   aGradient := aOutputGradient
   FOR i := 6 TO 1 STEP -1
      aGradient := ::BackwardEncode(aGradient)
   NEXT
   
   ::UpdateParameters(nLearningRate)
   
RETURN NIL

METHOD BackwardEncode(aGradient) CLASS Transformer
   LOCAL aFeedForwardGradient, aAttentionGradient
   
   // Verificar si las entradas guardadas son NIL
   IF ::aLastFeedForwardInput == NIL .OR. ::aLastAttentionInput == NIL
      ? "Error: Entradas guardadas en BackwardEncode son NIL"
      RETURN NIL
   ENDIF

   // Backward pass through Feed Forward
   aFeedForwardGradient := ::BackwardLayerNorm(aGradient, ::aLastFeedForwardInput)
   IF aFeedForwardGradient != NIL
      aFeedForwardGradient := ::BackwardFeedForward(aFeedForwardGradient, ::aLastFeedForwardInput)
   ENDIF

   // Add residual connection
   aGradient := ::ElementWiseAddition(aGradient, aFeedForwardGradient)

   // Backward pass through Multi-Head Attention
   aAttentionGradient := ::BackwardLayerNorm(aGradient, ::aLastAttentionInput)
   IF aAttentionGradient != NIL
      aAttentionGradient := ::BackwardMultiHeadAttention(aAttentionGradient, ::aLastAttentionInput, ::aLastAttentionInput, ::aLastAttentionInput)
   ENDIF

   // Add residual connection
   aGradient := ::ElementWiseAddition(aGradient, aAttentionGradient)

RETURN aGradient

METHOD BackwardMultiHeadAttention(aInputGradient, aQuery, aKey, aValue) CLASS Transformer
   LOCAL nHeadDim := ::nModelDim / ::nHeads
   LOCAL aHeadGradients := {}
   LOCAL i, aQGrad, aKGrad, aVGrad, aConcatenatedGrad, aProjectedGrad
   
   // Backward pass through output projection
   aProjectedGrad := ::BackwardLinearProjection(aInputGradient, NIL, "output")
   
   // Split gradient for each head
   aConcatenatedGrad := ::SplitHeads(aProjectedGrad)
   
   FOR i := 1 TO ::nHeads
      // Backward pass through dot-product attention
      aQGrad := ::BackwardDotProductAttention(aConcatenatedGrad[i], aQuery, aKey, aValue)
      aKGrad := ::BackwardDotProductAttention(aConcatenatedGrad[i], aQuery, aKey, aValue)
      aVGrad := ::BackwardDotProductAttention(aConcatenatedGrad[i], aQuery, aKey, aValue)
      
      // Backward pass through linear projections
      AAdd(aHeadGradients, ::BackwardLinearProjection(aQGrad, aQuery, "attention_q" + AllTrim(Str(i))))
      AAdd(aHeadGradients, ::BackwardLinearProjection(aKGrad, aKey, "attention_k" + AllTrim(Str(i))))
      AAdd(aHeadGradients, ::BackwardLinearProjection(aVGrad, aValue, "attention_v" + AllTrim(Str(i))))
   NEXT
   
RETURN ::SumGradients(aHeadGradients)

METHOD BackwardDotProductAttention(aGradient, aQuery, aKey, aValue) CLASS Transformer
   LOCAL aScoresGrad, aSoftmaxGrad, aQGrad, aKGrad, aVGrad
   LOCAL nDimK := Len(aKey[1])
   
   // Gradient of V
   aVGrad := ::MatMul(::Transpose(::SoftMax(::MatMul(aQuery, ::Transpose(aKey)))), aGradient)
   
   // Gradient of Softmax
   aSoftmaxGrad := ::BackwardSoftMax(aGradient, ::SoftMax(::MatMul(aQuery, ::Transpose(aKey))))
   
   // Gradient of scaling
   aScoresGrad := ::ElementWiseMultiplication(aSoftmaxGrad, Array(Len(aSoftmaxGrad), {|| Array(Len(aSoftmaxGrad[1]), {|| 1 / Sqrt(nDimK) })}))
   
   // Gradient of Q and K
   aQGrad := ::MatMul(aScoresGrad, aKey)
   aKGrad := ::MatMul(::Transpose(aScoresGrad), aQuery)
   
RETURN {aQGrad, aKGrad, aVGrad}

METHOD BackwardFeedForward(aInputGradient, aInput) CLASS Transformer
   LOCAL aHiddenGradient
   
   // Backward pass through second linear projection
   aHiddenGradient := ::BackwardLinearProjection(aInputGradient, NIL, "feedforward2")
   
   // Backward pass through ReLU
   aHiddenGradient := ::BackwardReLU(aHiddenGradient, NIL)  // We would need to store the input to ReLU in the forward pass
   
   // Backward pass through first linear projection
RETURN ::BackwardLinearProjection(aHiddenGradient, aInput, "feedforward1")

METHOD BackwardLayerNorm(aInputGradient, aInput) CLASS Transformer
   LOCAL aGammaGrad, aBetaGrad, aInputGrad
   LOCAL i, j, nMean, nVar
   
   aGammaGrad := Array(::nModelDim)
   aBetaGrad := Array(::nModelDim)
   aInputGrad := Array(Len(aInput))
   
   FOR i := 1 TO Len(aInput)
      nMean := ::Mean(aInput[i])
      nVar := ::Variance(aInput[i], nMean)
      aInputGrad[i] := Array(::nModelDim)
      
      FOR j := 1 TO ::nModelDim
         aGammaGrad[j] += aInputGradient[i][j] * (aInput[i][j] - nMean) / Sqrt(nVar + (1 * 10^-6) ) // 1e-6
         aBetaGrad[j] += aInputGradient[i][j]
         
         aInputGrad[i][j] := aInputGradient[i][j] * ::aGamma[j] / Sqrt(nVar + (1 * 10^-6) ) // 1e-6
      NEXT
   NEXT
   
   // Store gradients for gamma and beta
   AAdd(::aGradients, {"gamma", aGammaGrad})
   AAdd(::aGradients, {"beta", aBetaGrad})
   
RETURN aInputGrad

METHOD BackwardLinearProjection(aInputGradient, aInput, cType) CLASS Transformer
   LOCAL aWeightGrad, aBiasGrad, aInputGrad
   LOCAL i, j, nWeightIndex, nBiasIndex
   
   // Get predefined weights and biases
   nWeightIndex := AScan(::aWeights, {|a| Len(a[1]) == Len(aInputGradient[1])})
   nBiasIndex := AScan(::aBiases, {|a| Len(a) == Len(aInputGradient[1])})
   
   IF nWeightIndex == 0 .OR. nBiasIndex == 0
      // Handle error: weights or biases not found
      RETURN NIL
   ENDIF
   
   aWeightGrad := Array(Len(::aWeights[nWeightIndex]))
   aBiasGrad := Array(Len(::aBiases[nBiasIndex]))
   aInputGrad := Array(Len(aInput))
   
   // Compute gradients
   FOR i := 1 TO Len(aInput)
      aInputGrad[i] := Array(Len(aInput[i]))
      FOR j := 1 TO Len(aInputGradient[1])
         // Gradient for bias
         aBiasGrad[j] += aInputGradient[i][j]
         
         // Gradient for weights
         aWeightGrad[i][j] := aInput[i] * aInputGradient[i][j]
         
         // Gradient for input
         aInputGrad[i] := ::MatMul({aInputGradient[i]}, ::Transpose(::aWeights[nWeightIndex]))[1]
      NEXT
   NEXT
   
   // Store gradients
   AAdd(::aGradients, {cType + "_weight", aWeightGrad})
   AAdd(::aGradients, {cType + "_bias", aBiasGrad})
   
RETURN aInputGrad

METHOD BackwardReLU(aInputGradient, aInput) CLASS Transformer
   LOCAL aOutputGradient := Array(Len(aInput))
   LOCAL i, j
   
   FOR i := 1 TO Len(aInput)
      aOutputGradient[i] := Array(Len(aInput[i]))
      FOR j := 1 TO Len(aInput[i])
         IF aInput[i][j] > 0
            aOutputGradient[i][j] := aInputGradient[i][j]
         ELSE
            aOutputGradient[i][j] := 0
         ENDIF
      NEXT
   NEXT
   
RETURN aOutputGradient

METHOD BackwardSoftMax(aInputGradient, aInput) CLASS Transformer
   LOCAL aOutputGradient := Array(Len(aInput))
   LOCAL i, j, nSum
   
   FOR i := 1 TO Len(aInput)
      aOutputGradient[i] := Array(Len(aInput[i]))
      nSum := 0
      FOR j := 1 TO Len(aInput[i])
         nSum += aInput[i][j] * aInputGradient[i][j]
      NEXT
      FOR j := 1 TO Len(aInput[i])
         aOutputGradient[i][j] := aInput[i][j] * (aInputGradient[i][j] - nSum)
      NEXT
   NEXT
   
RETURN aOutputGradient

METHOD Encode(aInput) CLASS Transformer

   LOCAL aPositionalInput, aAttOutput, aFeedForwardOutput

   // Aplicar codificación posicional
   aPositionalInput := ::PositionalEncoding(aInput)
   
   // Almacenar la entrada para la atención de múltiples cabezas
   ::aLastAttentionInput := aPositionalInput

   // Realizar la atención de múltiples cabezas
   aAttOutput := ::MultiHeadAttention(aPositionalInput, aPositionalInput, aPositionalInput)
   
   // Aplicar normalización de capa y conexión residual
   aAttOutput := ::LayerNorm(::ElementWiseAddition(aPositionalInput, aAttOutput))
   
   // Almacenar la entrada para la capa feed-forward
   ::aLastFeedForwardInput := aAttOutput
   if( aPositionalInut == NIL )
      ? "Es NIL", InKey( 0 )
   endif   

   // Aplicar la capa feed-forward
   aFeedForwardOutput := ::FeedForward(aAttOutput)
   
   // Aplicar normalización de capa y conexión residual final
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

METHOD Forward( aInput ) CLASS Transformer

   LOCAL aAttOutput, aFeedForwardOutput

   aAttOutput := ::MultiHeadAttention( aInput, aInput, aInput ) 
   aAttOutput := ::LayerNorm( ::MatAdd( aInput, aAttOutput ) )

   aFeedForwardOutput := ::FeedForward( aAttOutput )
   aFeedForwardOutput := ::LayerNorm( ::MatAdd( aAttOutput, aFeedForwardOutput ) )

   RETURN aFeedForwardOutput

METHOD MultiHeadAttention( aQuery, aKey, aValue ) CLASS Transformer

   LOCAL nHeadDim := ::nModelDim / ::nHeads
   LOCAL aHeadOutputs := {}
   LOCAL i, aQ, aK, aV, aHeadOutput, aConcatenated, aProjected

   FOR i := 1 TO ::nHeads
      aQ := ::LinearProjection(aQuery, nHeadDim, "attention_q" + AllTrim(Str(i)))
      aK := ::LinearProjection(aKey, nHeadDim, "attention_k" + AllTrim(Str(i)))
      aV := ::LinearProjection(aValue, nHeadDim, "attention_v" + AllTrim(Str(i)))
      
      // ? "Query, key and value", hb_ValToExp( aQ ), hb_ValToExp( aK ), hb_ValToExp( aV )
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

METHOD LinearProjection( aInput, nOutputDim, cType ) CLASS Transformer

   LOCAL aOutput := {}
   LOCAL i, j, nWeightIndex, nBiasIndex

   // Get predefined weights and biases
   nWeightIndex := AScan(::aWeights, {|a| Len(a[1]) == nOutputDim})
   nBiasIndex := AScan(::aBiases, {|a| Len(a) == nOutputDim})

   IF nWeightIndex == 0 .OR. nBiasIndex == 0
       // Handle error: weights or biases not found
       RETURN NIL
   ENDIF

   // Perform matrix multiplication for the entire input
   ? "aInput", Len( aInput )
   ? "::aWeights[nWeightIndex]", Len( ::aWeights[ nWeightIndex ] )
   aOutput := ::MatMul( aInput, ::aWeights[ nWeightIndex ] )

   // Add biases
   FOR i := 1 TO Len( aOutput )
       FOR j := 1 TO nOutputDim
           aOutput[i][j] += ::aBiases[nBiasIndex][j]
       NEXT
   NEXT

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

METHOD SplitHeads(aInput) CLASS Transformer
   LOCAL nHeadDim := ::nModelDim / ::nHeads
   LOCAL aSplit := {}
   LOCAL i, j, k
   
   FOR i := 1 TO ::nHeads
      AAdd(aSplit, Array(Len(aInput)))
      FOR j := 1 TO Len(aInput)
         aSplit[i][j] := Array(nHeadDim)
         FOR k := 1 TO nHeadDim
            aSplit[i][j][k] := aInput[j][(i-1)*nHeadDim + k]
         NEXT
      NEXT
   NEXT
   
RETURN aSplit

METHOD SumGradients(aGradients) CLASS Transformer
   LOCAL aSumGradient
   LOCAL i, j, k
   
   // Initialize aSumGradient with the same structure as the first gradient in aGradients
   aSumGradient := AClone(aGradients[1])
   
   // Initialize all elements of aSumGradient to zero
   FOR i := 1 TO Len(aSumGradient)
      IF HB_ISARRAY(aSumGradient[i])
         FOR j := 1 TO Len(aSumGradient[i])
            aSumGradient[i][j] := 0
         NEXT
      ELSE
         aSumGradient[i] := 0
      ENDIF
   NEXT
   
   // Sum up all gradients
   FOR i := 1 TO Len(aGradients)
      FOR j := 1 TO Len(aGradients[i])
         IF HB_ISARRAY(aGradients[i][j])
            FOR k := 1 TO Len(aGradients[i][j])
               aSumGradient[j][k] += aGradients[i][j][k]
            NEXT
         ELSE
            aSumGradient[j] += aGradients[i][j]
         ENDIF
      NEXT
   NEXT
   
RETURN aSumGradient

METHOD LinearTransformation(aInput, nOutputDim) CLASS Transformer

   LOCAL aOutput
   LOCAL aWeights := GenerateRandomMatrix( Len( aInput[ 1 ] ), nOutputDim )
   LOCAL aBiases := GenerateRandomVector( nOutputDim )
   LOCAL i, j

   // Realizar la transformación lineal: output = input * weights + biases
   aOutput := ::MatMul(aInput, aWeights)

   // Añadir biases
   FOR i := 1 TO Len(aOutput)
      FOR j := 1 TO Len(aOutput[i])
         aOutput[i][j] += aBiases[j]
      NEXT
   NEXT

RETURN aOutput

METHOD MatAdd( aMatrix1, aMatrix2 ) CLASS Transformer
   LOCAL aResult := {}
   LOCAL i, j, nRows, nCols

   // Verificar que las matrices tienen las mismas dimensiones
   IF Len(aMatrix1) != Len(aMatrix2) .OR. Len(aMatrix1[1]) != Len(aMatrix2[1])
       ? "Error: Las matrices deben tener las mismas dimensiones para la suma"
       RETURN NIL
   ENDIF

   nRows := Len(aMatrix1)
   nCols := Len(aMatrix1[1])

   // Realizar la suma elemento por elemento
   FOR i := 1 TO nRows
       AAdd(aResult, Array(nCols))
       FOR j := 1 TO nCols
           aResult[i][j] := aMatrix1[i][j] + aMatrix2[i][j]
       NEXT
   NEXT

RETURN aResult

METHOD MatMul( aMatrix1, aMatrix2 ) CLASS Transformer

   LOCAL aResult := {}
   LOCAL i, j, k, nSum

   if Len( aMatrix1[ 1 ] ) != Len( aMatrix2 )
      ? "Matrices can't be multiplied", Len( aMatrix1[ 1 ] ), Len( aMatrix2 )
      return nil
   endif   

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

METHOD UpdateParameters(nLearningRate) CLASS Transformer
   LOCAL i, aGradient
   
   FOR i := 1 TO Len(::aGradients)
      aGradient := ::aGradients[i]
      DO CASE
      CASE "weight" $ aGradient[1]
         ::UpdateWeights(aGradient[1], aGradient[2], nLearningRate)
      CASE "bias" $ aGradient[1]
         ::UpdateBiases(aGradient[1], aGradient[2], nLearningRate)
      CASE aGradient[1] == "gamma"
         ::UpdateGamma(aGradient[2], nLearningRate)
      CASE aGradient[1] == "beta"
         ::UpdateBeta(aGradient[2], nLearningRate)
      ENDCASE
   NEXT
   
RETURN NIL

METHOD UpdateWeights(cKey, aGradient, nLearningRate) CLASS Transformer
   LOCAL nIndex := AScan(::aWeights, {|a| Len(a) == Len(aGradient) .AND. Len(a[1]) == Len(aGradient[1])})
   LOCAL i, j
   
   IF nIndex > 0
      FOR i := 1 TO Len(::aWeights[nIndex])
         FOR j := 1 TO Len(::aWeights[nIndex][i])
            ::aWeights[nIndex][i][j] -= nLearningRate * aGradient[i][j]
         NEXT
      NEXT
   ENDIF
   
RETURN NIL

METHOD UpdateBiases(cKey, aGradient, nLearningRate) CLASS Transformer
   LOCAL nIndex := AScan(::aBiases, {|a| Len(a) == Len(aGradient)})
   LOCAL i
   
   IF nIndex > 0
      FOR i := 1 TO Len(::aBiases[nIndex])
         ::aBiases[nIndex][i] -= nLearningRate * aGradient[i]
      NEXT
   ENDIF
   
RETURN NIL

METHOD UpdateGamma(aGradient, nLearningRate) CLASS Transformer
   LOCAL i
   
   FOR i := 1 TO Len(::aGamma)
      ::aGamma[i] -= nLearningRate * aGradient[i]
   NEXT
   
RETURN NIL

METHOD UpdateBeta(aGradient, nLearningRate) CLASS Transformer
   LOCAL i
   
   FOR i := 1 TO Len(::aBeta)
      ::aBeta[i] -= nLearningRate * aGradient[i]
   NEXT
   
RETURN NIL

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
   LOCAL nMean, nVariance, nEpsilon := ( 1 * 10^-8 ) // 1e-8
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

FUNCTION GenerateRandomMatrix(nRows, nCols)
   LOCAL aMatrix := Array(nRows, nCols), i, j
   FOR i := 1 TO nRows
       FOR j := 1 TO nCols
           aMatrix[i,j] := hb_Random(-1, 1)
       NEXT    
   NEXT
RETURN aMatrix

FUNCTION GenerateRandomVector(nSize)
   LOCAL aVector := Array(nSize), i
   FOR i := 1 TO nSize
       aVector[i] := hb_Random(-1, 1)
   NEXT
RETURN aVector

