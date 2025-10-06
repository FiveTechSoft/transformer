#include "hbclass.ch"

/*
* Clase: TransformerEncoderBlock
* -------------------------------
* Implementa un único bloque Encoder, capaz de realizar el forward pass,
* backpropagation, y actualizar sus propios pesos.
* CORRECCIONES: Cache para pre-norms en LayerNorm, locals en Backward, escalado en atención,
* y momentos de Adam no reseteados en ZeroGrads.
*/
CLASS TransformerEncoderBlock
   // Pesos
   DATA oWq, oWk, oV, oW1, ob1, oW2, ob2, oGamma1, oBeta1, oGamma2, oBeta2
   // Gradientes
   DATA gWq, gWk, gV, gW1, gb1, gW2, gb2, gGamma1, gBeta1, gGamma2, gBeta2
   // Momentos de Adam (m y v)
   DATA mM_Wq, mV_Wq, mM_Wk, mV_Wk, mM_V, mV_V, mM_W1, mV_W1, mM_b1, mV_b1
   DATA mM_W2, mV_W2, mM_b2, mV_b2, mM_Gamma1, mV_Gamma1, mM_Beta1, mV_Beta1
   DATA mM_Gamma2, mV_Gamma2, mM_Beta2, mV_Beta2
   // Cache (AGREGADO: cPreNorm1 y cPreNorm2 para LayerNorm backward)
   DATA cInput, cNormalized1, cActivated, cQ, cK, cV, cAttentionWeights, cPreNorm1, cPreNorm2
   // Otros
   DATA nInputDim, nHiddenDim, nHeadDim, nTimeStep

   METHOD New( nInputDim, nHiddenDim, nHeadDim ) CONSTRUCTOR
   METHOD Forward( mInput )
   METHOD Backward( mDOutput )
   METHOD ZeroGrads()
   METHOD Update( nLr )
ENDCLASS

/*
* CONSTRUCTOR (CORREGIDO: Inicializa momentos a cero explícitamente)
*/
METHOD New( nInputDim, nHiddenDim, nHeadDim ) CLASS TransformerEncoderBlock
   ::nInputDim  := nInputDim
   ::nHiddenDim := nHiddenDim
   ::nHeadDim   := nHeadDim  // Asumir nHeadDim == nInputDim para suma residual
   ::nTimeStep  := 0

   // Inicializar Pesos
   ::oWq := HB_MATRIXRANDOM( nInputDim, nHeadDim )
   ::oWk := HB_MATRIXRANDOM( nInputDim, nHeadDim )
   ::oV  := HB_MATRIXRANDOM( nInputDim, nHeadDim )
   ::oW1 := HB_MATRIXRANDOM( nInputDim, nHiddenDim )
   ::ob1 := HB_MATRIXZERO( 1, nHiddenDim )
   ::oW2 := HB_MATRIXRANDOM( nHiddenDim, nInputDim )
   ::ob2 := HB_MATRIXZERO( 1, nInputDim )
   ::oGamma1 := HB_MATRIXFILL( HB_MATRIXZERO( 1, nInputDim ), 1.0 )
   ::oBeta1  := HB_MATRIXZERO( 1, nInputDim )
   ::oGamma2 := HB_MATRIXFILL( HB_MATRIXZERO( 1, nInputDim ), 1.0 )
   ::oBeta2  := HB_MATRIXZERO( 1, nInputDim )

   // CORREGIDO: Inicializar gradientes y momentos a cero
   ::ZeroGrads()  // Solo grads; momentos se inicializan aquí
   ::mM_Wq := HB_MATRIXZERO( nInputDim, nHeadDim ); ::mV_Wq := HB_MATRIXZERO( nInputDim, nHeadDim )
   ::mM_Wk := HB_MATRIXZERO( nInputDim, nHeadDim ); ::mV_Wk := HB_MATRIXZERO( nInputDim, nHeadDim )
   ::mM_V  := HB_MATRIXZERO( nInputDim, nHeadDim ); ::mV_V  := HB_MATRIXZERO( nInputDim, nHeadDim )
   ::mM_W1 := HB_MATRIXZERO( nInputDim, nHiddenDim ); ::mV_W1 := HB_MATRIXZERO( nInputDim, nHiddenDim )
   ::mM_b1 := HB_MATRIXZERO( 1, nHiddenDim ); ::mV_b1 := HB_MATRIXZERO( 1, nHiddenDim )
   ::mM_W2 := HB_MATRIXZERO( nHiddenDim, nInputDim ); ::mV_W2 := HB_MATRIXZERO( nHiddenDim, nInputDim )
   ::mM_b2 := HB_MATRIXZERO( 1, nInputDim ); ::mV_b2 := HB_MATRIXZERO( 1, nInputDim )
   ::mM_Gamma1 := HB_MATRIXZERO( 1, nInputDim ); ::mV_Gamma1 := HB_MATRIXZERO( 1, nInputDim )
   ::mM_Beta1 := HB_MATRIXZERO( 1, nInputDim ); ::mV_Beta1 := HB_MATRIXZERO( 1, nInputDim )
   ::mM_Gamma2 := HB_MATRIXZERO( 1, nInputDim ); ::mV_Gamma2 := HB_MATRIXZERO( 1, nInputDim )
   ::mM_Beta2 := HB_MATRIXZERO( 1, nInputDim ); ::mV_Beta2 := HB_MATRIXZERO( 1, nInputDim )

RETURN Self

/*
* ZeroGrads() (CORREGIDO: Solo resetea gradientes, NO momentos de Adam)
*/
METHOD ZeroGrads() CLASS TransformerEncoderBlock
   // Solo poner a cero los gradientes
   ::gWq := HB_MATRIXZERO( ::nInputDim, ::nHeadDim ); ::gWk := HB_MATRIXZERO( ::nInputDim, ::nHeadDim ); ::gV := HB_MATRIXZERO( ::nInputDim, ::nHeadDim )
   ::gW1 := HB_MATRIXZERO( ::nInputDim, ::nHiddenDim ); ::gb1 := HB_MATRIXZERO( 1, ::nHiddenDim )
   ::gW2 := HB_MATRIXZERO( ::nHiddenDim, ::nInputDim ); ::gb2 := HB_MATRIXZERO( 1, ::nInputDim )
   ::gGamma1 := HB_MATRIXZERO( 1, ::nInputDim ); ::gBeta1 := HB_MATRIXZERO( 1, ::nInputDim )
   ::gGamma2 := HB_MATRIXZERO( 1, ::nInputDim ); ::gBeta2 := HB_MATRIXZERO( 1, ::nInputDim )
   // Los momentos NO se resetean aquí; persisten para Adam
RETURN Nil

/*
* Forward( mInput ) (CORREGIDO: Cachear pre-norms para backward)
*/
METHOD Forward( mInput ) CLASS TransformerEncoderBlock
   LOCAL mAttentionOutput, mNormalized1, mFFN_Output, mEncoderOutput  // Locales existentes
   LOCAL mScores, mSublayer1, mLinear1, mWithBias1, mLinear2  // AGREGADOS: Locales faltantes

   // Guardar la entrada para el backward pass
   ::cInput := mInput

   // 1. Self-Attention
   ::cQ := HB_MATRIXMULTIPLY( mInput, ::oWq )
   ::cK := HB_MATRIXMULTIPLY( mInput, ::oWk )
   ::cV := HB_MATRIXMULTIPLY( mInput, ::oV )
   mScores := HB_MATRIXMULTIPLY( ::cQ, HB_MATRIXTRANSPOSE(::cK) )
   mScores := HB_MATRIXDIVSCALAR( mScores, Sqrt(::nHeadDim) )
   ::cAttentionWeights := HB_SOFTMAX( mScores )
   mAttentionOutput  := HB_MATRIXMULTIPLY( ::cAttentionWeights, ::cV )

   // 2. Add & Norm 1 (CORREGIDO: Cachear pre-norm)
   mSublayer1   := HB_MATRIXADD( mInput, mAttentionOutput )
   ::cPreNorm1  := mSublayer1  // AGREGADO: Cache para backward
   ::cNormalized1 := HB_LAYERNORM( mSublayer1, ::oGamma1, ::oBeta1 )

   // 3. Feed-Forward Network
   mLinear1   := HB_MATRIXMULTIPLY( ::cNormalized1, ::oW1 )
   mWithBias1 := HB_MATRIXADDBROADCAST( mLinear1, ::ob1 )
   ::cActivated := HB_RELU( mWithBias1 )
   mLinear2   := HB_MATRIXMULTIPLY( ::cActivated, ::oW2 )
   mFFN_Output := HB_MATRIXADDBROADCAST( mLinear2, ::ob2 )

   // 4. Add & Norm 2 (CORREGIDO: Cachear pre-norm)
   mSublayer2   := HB_MATRIXADD( ::cNormalized1, mFFN_Output )  // Nota: Residual desde normalized1
   ::cPreNorm2  := mSublayer2  // AGREGADO: Cache para backward
   mEncoderOutput := HB_LAYERNORM( mSublayer2, ::oGamma2, ::oBeta2 )

RETURN mEncoderOutput

/*
* Backward( mDOutput ) (CORREGIDO: Locales completos, cache pre-norms, escalado atención, residual fijo)
*/
METHOD Backward( mDOutput ) CLASS TransformerEncoderBlock
   LOCAL aGrads, mDNormalized1, mDFFN_Output, mDInput  // Locales existentes
   LOCAL mDLinear2, mDActivated, mDReluInput, mDLinear1, mDAttentionOutput, mDInput_from_res1  // AGREGADOS
   LOCAL mDAttentionWeights, mDV, mDScores, mDQ, mDK  // AGREGADOS
   LOCAL mDInput_from_Q, mDInput_from_K, mDInput_from_V  // AGREGADOS

   // --- Backprop a través de Add & Norm 2 ---
   aGrads         := HB_LAYERNORM_BACKWARD( mDOutput, ::cPreNorm2, ::oGamma2, ::oBeta2 )  // CORREGIDO: Usa cPreNorm2
   mDFFN_Output   := aGrads[1]
   ::gGamma2      := HB_MATRIXADD( ::gGamma2, aGrads[2] )
   ::gBeta2       := HB_MATRIXADD( ::gBeta2, aGrads[3] )
   mDNormalized1  := mDFFN_Output  // Gradiente de la conexión residual (desde FFN a normalized1)

   // --- Backprop a través de Feed-Forward ---
   aGrads      := HB_MATRIXADDBROADCAST_BACKWARD( mDFFN_Output )
   mDLinear2   := aGrads[1]
   ::gb2       := HB_MATRIXADD( ::gb2, aGrads[2] )
   aGrads      := HB_MATRIXMULTIPLY_BACKWARD( mDLinear2, ::cActivated, ::oW2 )
   mDActivated := aGrads[1]
   ::gW2       := HB_MATRIXADD( ::gW2, aGrads[2] )
   mDReluInput := HB_RELU_BACKWARD( mDActivated, ::cActivated )  // Usa cache ::cActivated
   aGrads      := HB_MATRIXADDBROADCAST_BACKWARD( mDReluInput )
   mDLinear1   := aGrads[1]
   ::gb1       := HB_MATRIXADD( ::gb1, aGrads[2] )
   aGrads      := HB_MATRIXMULTIPLY_BACKWARD( mDLinear1, ::cNormalized1, ::oW1 )
   mDNormalized1 := HB_MATRIXADD( mDNormalized1, aGrads[1] )  // CORREGIDO: Sumar grad de FFN a mDNormalized1
   ::gW1         := HB_MATRIXADD( ::gW1, aGrads[2] )

   // --- Backprop a través de Add & Norm 1 ---
   aGrads            := HB_LAYERNORM_BACKWARD( mDNormalized1, ::cPreNorm1, ::oGamma1, ::oBeta1 )  // CORREGIDO: Usa cPreNorm1
   mDAttentionOutput := aGrads[1]
   ::gGamma1         := HB_MATRIXADD( ::gGamma1, aGrads[2] )
   ::gBeta1          := HB_MATRIXADD( ::gBeta1, aGrads[3] )
   mDInput_from_res1 := mDAttentionOutput  // Grad residual de atención a input

   // --- Backprop a través de Self-Attention ---
   aGrads             := HB_MATRIXMULTIPLY_BACKWARD( mDAttentionOutput, ::cAttentionWeights, ::cV )
   mDAttentionWeights := aGrads[1]
   mDV                := aGrads[2]
   mDScores           := HB_SOFTMAXBACKWARD( mDAttentionWeights, ::cAttentionWeights )
   mDScores           := HB_MATRIXMULSCALAR( mDScores, 1 / Sqrt( ::nHeadDim ) )  // CORREGIDO: Escalar por 1/sqrt(d_k)
   aGrads             := HB_MATRIXMULTIPLY_BACKWARD( mDScores, ::cQ, HB_MATRIXTRANSPOSE(::cK) )
   mDQ                := aGrads[1]
   mDK                := HB_MATRIXTRANSPOSE(aGrads[2])

   // Backprop final a los pesos Q, K, V
   aGrads         := HB_MATRIXMULTIPLY_BACKWARD( mDQ, ::cInput, ::oWq )
   mDInput_from_Q := aGrads[1]
   ::gWq          := HB_MATRIXADD( ::gWq, aGrads[2] )
   aGrads         := HB_MATRIXMULTIPLY_BACKWARD( mDK, ::cInput, ::oWk )
   mDInput_from_K := aGrads[1]
   ::gWk          := HB_MATRIXADD( ::gWk, aGrads[2] )
   aGrads         := HB_MATRIXMULTIPLY_BACKWARD( mDV, ::cInput, ::oV )
   mDInput_from_V := aGrads[1]
   ::gV           := HB_MATRIXADD( ::gV, aGrads[2] )

   // Sumar todos los gradientes que llegan a la entrada original
   mDInput := HB_MATRIXADD( mDInput_from_res1, HB_MATRIXADD( mDInput_from_Q, HB_MATRIXADD( mDInput_from_K, mDInput_from_V ) ) )

RETURN mDInput

/*
* Update( nLr ) (Sin cambios: Usa HB_ADAMUPDATE correctamente)
*/
METHOD Update( nLr ) CLASS TransformerEncoderBlock
   ::nTimeStep++
   HB_ADAMUPDATE( ::oWq, ::gWq, ::mM_Wq, ::mV_Wq, ::nTimeStep, nLr, 0.9, 0.999, 0.00000001 )
   HB_ADAMUPDATE( ::oWk, ::gWk, ::mM_Wk, ::mV_Wk, ::nTimeStep, nLr, 0.9, 0.999, 0.00000001 )
   HB_ADAMUPDATE( ::oV,  ::gV,  ::mM_V,  ::mV_V,  ::nTimeStep, nLr, 0.9, 0.999, 0.00000001 )
   HB_ADAMUPDATE( ::oW1, ::gW1, ::mM_W1, ::mV_W1, ::nTimeStep, nLr, 0.9, 0.999, 0.00000001 )
   HB_ADAMUPDATE( ::ob1, ::gb1, ::mM_b1, ::mV_b1, ::nTimeStep, nLr, 0.9, 0.999, 0.00000001 )
   HB_ADAMUPDATE( ::oW2, ::gW2, ::mM_W2, ::mV_W2, ::nTimeStep, nLr, 0.9, 0.999, 0.00000001 )
   HB_ADAMUPDATE( ::ob2, ::gb2, ::mM_b2, ::mV_b2, ::nTimeStep, nLr, 0.9, 0.999, 0.00000001 )
   HB_ADAMUPDATE( ::oGamma1, ::gGamma1, ::mM_Gamma1, ::mV_Gamma1, ::nTimeStep, nLr, 0.9, 0.999, 0.00000001 )
   HB_ADAMUPDATE( ::oBeta1,  ::gBeta1,  ::mM_Beta1,  ::mV_Beta1,  ::nTimeStep, nLr, 0.9, 0.999, 0.00000001 )
   HB_ADAMUPDATE( ::oGamma2, ::gGamma2, ::mM_Gamma2, ::mV_Gamma2, ::nTimeStep, nLr, 0.9, 0.999, 0.00000001 )
   HB_ADAMUPDATE( ::oBeta2,  ::gBeta2,  ::mM_Beta2,  ::mV_Beta2,  ::nTimeStep, nLr, 0.9, 0.999, 0.00000001 )
RETURN Nil

/*
* Clase TransformerModel: Gestiona una pila de bloques Encoder.
*/
CLASS TransformerModel
   DATA aEncoderBlocks, nLayers, oOutputProj, oOutputProjGrad, mLastBlockOutput
   DATA oOutputProj_m, oOutputProj_v, nTimeStep
   DATA oEmbeddings, oEmbeddingsGrad, oEmbeddings_m, oEmbeddings_v, nVocabSize, nEmbedDim
   METHOD New( nLayers, nInputDim, nHiddenDim, nHeadDim, nVocabSize ) CONSTRUCTOR
   METHOD Forward( mInput )
   METHOD Backward( mDOutput )
   METHOD ZeroGrads()
   METHOD Update( nLr )
ENDCLASS

METHOD New( nLayers, nInputDim, nHiddenDim, nHeadDim, nVocabSize ) CLASS TransformerModel
   LOCAL i
   ::nLayers := nLayers
   ::nVocabSize := nVocabSize
   ::nEmbedDim := nInputDim  // Assuming embed dim == input dim
   ::aEncoderBlocks := {}
   FOR i := 1 TO nLayers
      AAdd( ::aEncoderBlocks, TransformerEncoderBlock():New( nInputDim, nHiddenDim, nHeadDim ) )
   NEXT
   ::oOutputProj := HB_MATRIXRANDOM( nInputDim, nVocabSize )
   ::oOutputProjGrad := HB_MATRIXZERO( nInputDim, nVocabSize )
   ::oOutputProj_m := HB_MATRIXZERO( nInputDim, nVocabSize )
   ::oOutputProj_v := HB_MATRIXZERO( nInputDim, nVocabSize )
   ::oEmbeddings := HB_MATRIXRANDOM( nVocabSize, nInputDim )
   ::oEmbeddingsGrad := HB_MATRIXZERO( nVocabSize, nInputDim )
   ::oEmbeddings_m := HB_MATRIXZERO( nVocabSize, nInputDim )
   ::oEmbeddings_v := HB_MATRIXZERO( nVocabSize, nInputDim )
   ::mLastBlockOutput := Nil
   ::nTimeStep := 0
RETURN Self

METHOD ZeroGrads() CLASS TransformerModel
   AEval( ::aEncoderBlocks, {|oBlock| oBlock:ZeroGrads()} )
   ::oOutputProjGrad := HB_MATRIXZERO( Len(::oOutputProjGrad), Len(::oOutputProjGrad[1]) )
   ::oEmbeddingsGrad := HB_MATRIXZERO( ::nVocabSize, ::nEmbedDim )
RETURN Nil

METHOD Update( nLr ) CLASS TransformerModel
   AEval( ::aEncoderBlocks, {|oBlock| oBlock:Update(nLr)} )
   ::nTimeStep++
   HB_ADAMUPDATE( ::oOutputProj, ::oOutputProjGrad, ::oOutputProj_m, ::oOutputProj_v, ::nTimeStep, nLr, 0.9, 0.999, 0.00000001 )
   HB_ADAMUPDATE( ::oEmbeddings, ::oEmbeddingsGrad, ::oEmbeddings_m, ::oEmbeddings_v, ::nTimeStep, nLr, 0.9, 0.999, 0.00000001 )
RETURN Nil

METHOD Forward( mInput ) CLASS TransformerModel
   LOCAL mLastBlockOutput
   AEval( ::aEncoderBlocks, {|oBlock| mInput := oBlock:Forward(mInput)} )
   mLastBlockOutput := mInput
   mInput := HB_MATRIXMULTIPLY( mInput, ::oOutputProj )
   ::mLastBlockOutput := mLastBlockOutput  // Save for backward
RETURN mInput

METHOD Backward( mDOutput ) CLASS TransformerModel
   LOCAL i
   // Backward through output projection
   ::oOutputProjGrad := HB_MATRIXADD( ::oOutputProjGrad, HB_MATRIXMULTIPLY( HB_MATRIXTRANSPOSE(::mLastBlockOutput), mDOutput ) )
   mDOutput := HB_MATRIXMULTIPLY( mDOutput, HB_MATRIXTRANSPOSE(::oOutputProj) )
   // Backward through blocks
   FOR i := ::nLayers TO 1 STEP -1
      mDOutput := ::aEncoderBlocks[i]:Backward( mDOutput )
   NEXT
RETURN mDOutput

// ========================================================================
// SECCIÓN DE FUNCIONES AUXILIARES
// ========================================================================

FUNCTION CreatePositionalEncoding( nSeqLen, nEmbedDim )
   LOCAL mPE := HB_MATRIXZERO( nSeqLen, nEmbedDim )
   LOCAL pos, i, nAngle, pRow
   FOR pos := 1 TO nSeqLen
      pRow := mPE[pos]
      FOR i := 1 TO nEmbedDim STEP 2
         nAngle := (pos - 1) / ( 10000 ^ ( (i - 1) / nEmbedDim ) )
         pRow[i]   := Sin( nAngle )
         IF (i + 1) <= nEmbedDim
            pRow[i+1] := Cos( nAngle )
         ENDIF
      NEXT
   NEXT
RETURN mPE

STATIC FUNCTION CreateOneHotEmbeddings( nVocabSize, nEmbedDim )
   LOCAL mEmbeddings := {}
   LOCAL i, aRow
   // Asegurarse de que la dimensión del embedding sea suficiente
   IF nEmbedDim < nVocabSize
      nEmbedDim := nVocabSize
   ENDIF
   FOR i := 0 TO nVocabSize - 1
      aRow := Array( nEmbedDim, 0 )
      aRow[ i + 1 ] := 1
      AAdd( mEmbeddings, aRow )
   NEXT
RETURN mEmbeddings

STATIC FUNCTION CreateMatrixFromTokens( aTokens, mEmbeddings )
   LOCAL mMatrix := {}
   AEval( aTokens, {|nToken| AAdd( mMatrix, mEmbeddings[ nToken + 1 ] ) } )
RETURN mMatrix

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
   LOCAL nSumSq := 0, i
   FOR i := 1 TO Len(aVec1)
      nSumSq += (aVec1[i] - aVec2[i])^2
   NEXT
RETURN nSumSq