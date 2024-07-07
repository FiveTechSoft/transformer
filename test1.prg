PROCEDURE Main()

   local oTransformer, aInput, aOutput
   
   // AltD( 1 )
   // AltD()

   // Inicializar el Transformer
   oTransformer := Transformer():New( 8, 64, 64, 100 )

   ? "==== 0 ===="
   InKey( 0 )

   // Crear un input de ejemplo (en una implementación real, esto sería una secuencia de tokens)
   aInput := GenerateRandomMatrix( 10, 64 )

   ? "==== 1 ===="
   InKey( 0 )
   
   // Procesar el input a través del Transformer
   aOutput := oTransformer:Forward( aInput )
   
   ? "==== 2 ===="
   InKey( 0 )

   // Imprimir algunos resultados de ejemplo
   ? "Dimensiones del output:", Len( aOutput ), "x", Len( aOutput[ 1 ] )
   // ? "Primeros 5 valores del primer vector del output:"
   // ? hb_ValToExp( aOutput )

RETURN
