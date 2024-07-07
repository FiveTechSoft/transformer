PROCEDURE Main()
   
   LOCAL oTransformer
   LOCAL aInput
   LOCAL aOutput
   
   // Inicializar el Transformer
   oTransformer := Transformer():New( 8, 512, 2048, 128 )
   
   // Crear un input de ejemplo (en una implementación real, esto sería una secuencia de tokens)
   aInput := Array(10, 512)
   AEval(aInput, {|x| AFill(x, 1)})
   
   // Procesar el input a través del Transformer
   aOutput := oTransformer:Forward(aInput)
   
   // Imprimir algunos resultados de ejemplo
   ? "Dimensiones del output:", Len(aOutput), "x", Len(aOutput[1])
   ? "Primeros 5 valores del primer vector del output:"
   AEval(aOutput[1], {|x, i| if(i <= 5, QOut(x), NIL)})

RETURN
