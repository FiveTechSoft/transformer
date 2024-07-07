PROCEDURE Main()

   local oTransformer, aInput, aOutput
   
   // Inicializar el Transformer
   oTransformer := Transformer():New( 4, 64, 64, 64 )
   
   // Crear un input de ejemplo (en una implementación real, esto sería una secuencia de tokens)
   aInput := GenerateRandomMatrix( 10, 512 )
   
   // Procesar el input a través del Transformer
   aOutput := oTransformer:Forward( aInput )
   
   // Imprimir algunos resultados de ejemplo
   ? "Dimensiones del output:", Len( aOutput ), "x", Len( aOutput[ 1 ] )
   ? "Primeros 5 valores del primer vector del output:"
   ? hb_ValToExp( aOutput )

RETURN
