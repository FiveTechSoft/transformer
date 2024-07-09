#include "hbclass.ch"

// Función principal
FUNCTION Main()
    LOCAL oTransformer
    LOCAL aTrainingData := { ;
        "el gato duerme en la cama por la noche en su", ;
        "el perro juega en el parque", ;
        "la niña lee un libro" ;
    }
    LOCAL aVocab := {}
    LOCAL aTokenizedData := {}
    LOCAL aOneHotData := {}
    LOCAL i, j, aInput, aOutput, nEpochs := 100
    LOCAL cTestPhrase := "el gato juega", cPhrase, cToken
    LOCAL aTestTokens, aTestOneHot, aGenerated, aPhrase
    
    // Crear vocabulario
    FOR EACH cPhrase IN aTrainingData
      for each cToken in hb_ATokens( cPhrase, " " )
         if AScan(aVocab, cToken) == 0 
            AAdd(aVocab, cToken)
         endif   
      next  
    NEXT
    
    // Tokenizar y convertir a one-hot
    FOR EACH cPhrase IN aTrainingData
        aTokenizedData := Tokenize(cPhrase)
        AAdd(aOneHotData, TokensToOneHot(aTokenizedData, aVocab))
    NEXT
    
    // Crear y entrenar el transformer
    oTransformer := Transformer():New( 4, 16, 256, 10 )  // 4 heads, 12 model dim, 256 ff dim, max 10 tokens
    
    // Entrenamiento
    FOR i := 1 TO nEpochs
        FOR EACH aPhrase IN aOneHotData
            // Usar los primeros n-1 tokens como entrada y el último como salida
            aInput := AClone(aPhrase)
            aOutput := ATail(aPhrase)
            ADel(aInput, Len(aInput))
            aSize(aInput, Len(aInput) - 1)
            
            // Forward pass
            aGenerated := oTransformer:Forward( aInput )
            
            // Backward pass (asumimos que el último token es la salida deseada)
            oTransformer:Backward({aOutput}, 0.01)  // Learning rate de 0.01
        NEXT
    NEXT
    
    // Prueba
    aTestTokens := Tokenize(cTestPhrase)
    aTestOneHot := TokensToOneHot(aTestTokens, aVocab)
    
    aGenerated := oTransformer:Forward(aTestOneHot)
    
    // Convertir la salida a token
    ? hb_ValToExp( aGenerated )
    // nMaxIndex := AScan(ATail(aGenerated), {|x| x == hb_Max(ATail(aGenerated))})
    // cGeneratedWord := aVocab[nMaxIndex]
    
    // ? "Frase de prueba:", cTestPhrase
    // ? "Palabra generada:", cGeneratedWord

RETURN NIL

// Función auxiliar para tokenizar texto
FUNCTION Tokenize(cText)
RETURN hb_ATokens(Lower(cText), " ")

// Función auxiliar para convertir tokens a one-hot encoding
FUNCTION TokensToOneHot(aTokens, aVocab)
LOCAL aOneHot := {}
LOCAL nVocabSize := Len(aVocab)
LOCAL nIndex, cToken

FOR EACH cToken IN aTokens
    nIndex := AScan(aVocab, cToken)
    IF nIndex > 0
        AAdd(aOneHot, AFill( Array(nVocabSize), 0 ) )
        aOneHot[ Len( aOneHot ) ][ nIndex ] := 1
    ENDIF
NEXT

RETURN aOneHot
