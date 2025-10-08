// ========================================================================
// matrixes.c
// Biblioteca de C para operaciones matriciales y de redes neuronales
// para su uso con Harbour.
// Versión: 2.0 (Refactorizada con patrón Worker/Wrapper)
// ========================================================================

#include <hbapi.h>
#include <hbapiitm.h>
#include <hbapierr.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#ifndef M_E
   #define M_E 2.71828182845904523536
#endif

// ========================================================================
// SECCIÓN 1: PROTOTIPOS DE FUNCIONES
// ========================================================================

// --- Prototipos de los "Workers" (Lógica interna en C) ---
static PHB_ITEM matrix_clone( PHB_ITEM pMatrix );
static PHB_ITEM matrix_zero( HB_SIZE nRows, HB_SIZE nCols );
static PHB_ITEM matrix_random( HB_SIZE nRows, HB_SIZE nCols );
static PHB_ITEM matrix_transpose( PHB_ITEM pMatrix );
static PHB_ITEM matrix_multiply( PHB_ITEM pMatrix1, PHB_ITEM pMatrix2 );
static PHB_ITEM matrix_add( PHB_ITEM pMatrix1, PHB_ITEM pMatrix2 );
static PHB_ITEM matrix_sub( PHB_ITEM pMatrix1, PHB_ITEM pMatrix2 );
static PHB_ITEM matrix_elem_mult( PHB_ITEM pMatrix1, PHB_ITEM pMatrix2 );
static PHB_ITEM matrix_mul_scalar( PHB_ITEM pMatrix, double scalar );
static PHB_ITEM matrix_div_scalar( PHB_ITEM pMatrix, double scalar );
static PHB_ITEM matrix_sum_axis0( PHB_ITEM pMatrix );
static PHB_ITEM relu_forward( PHB_ITEM pMatrix );

// --- Prototipos de los "Wrappers" (API para Harbour) ---
HB_FUNC( HB_SEEDRAND );
HB_FUNC( HB_MATRIXCLONE );
HB_FUNC( HB_MATRIXZERO );
HB_FUNC( HB_MATRIXRANDOM );
HB_FUNC( HB_MATRIXTRANSPOSE );
HB_FUNC( HB_MATRIXMULTIPLY );
HB_FUNC( HB_MATRIXADD );
HB_FUNC( HB_MATRIXSUB );
HB_FUNC( HB_MATRIXELEMMULT );
HB_FUNC( HB_MATRIXMULSCALAR );
HB_FUNC( HB_MATRIXDIVSCALAR );
HB_FUNC( HB_MATRIXSUMAXIS0 );
HB_FUNC( HB_RELU );
HB_FUNC( HB_SOFTMAX );
HB_FUNC( HB_SGD_UPDATE );
HB_FUNC( HB_ADAMUPDATE );

// --- Prototipos de Backpropagation (API para Harbour) ---
HB_FUNC( HB_DROPOUT );
HB_FUNC( HB_DROPOUT_BACKWARD );
HB_FUNC( HB_SOFTMAXBACKWARD );
HB_FUNC( HB_RELU_BACKWARD );
HB_FUNC( HB_MATRIXMULTIPLY_BACKWARD );
HB_FUNC( HB_MATRIXADDBROADCAST_BACKWARD );
HB_FUNC( HB_LAYERNORM ); // Forward pass de LayerNorm
HB_FUNC( HB_LAYERNORM_BACKWARD );
HB_FUNC( HB_CROSSENTROPYLOSS );


// ========================================================================
// SECCIÓN 2: IMPLEMENTACIÓN DE LOS WORKERS (LÓGICA INTERNA)
// ========================================================================

// --- Workers de Creación y Utilidades ---

static PHB_ITEM matrix_clone( PHB_ITEM pMatrix )
{
    if( pMatrix && HB_IS_ARRAY( pMatrix ) ) {
        return hb_itemClone( pMatrix );
    }
    return hb_itemArrayNew(0);
}

static PHB_ITEM matrix_zero( HB_SIZE nRows, HB_SIZE nCols )
{
   HB_SIZE i, j;
   PHB_ITEM pMatrix, pRow;
   if( nRows > 0 && nCols > 0 )
   {
      pMatrix = hb_itemArrayNew( nRows );
      for( i = 0; i < nRows; i++ )
      {
         pRow = hb_itemArrayNew( nCols );
         for( j = 0; j < nCols; j++ ) { hb_arraySetND( pRow, j + 1, 0.0 ); }
         hb_arraySet( pMatrix, i + 1, pRow );
         hb_itemRelease( pRow );
      }
      return pMatrix;
   }
   return hb_itemArrayNew(0);
}

static PHB_ITEM matrix_random( HB_SIZE nRows, HB_SIZE nCols )
{
   HB_SIZE i, j;
   PHB_ITEM pMatrix, pRow;
   if( nRows > 0 && nCols > 0 )
   {
      pMatrix = hb_itemArrayNew( nRows );
      for( i = 0; i < nRows; i++ )
      {
         pRow = hb_itemArrayNew( nCols );
         for( j = 0; j < nCols; j++ )
         {
            double randomValue = ( (double)rand() / RAND_MAX ) - 0.5;
            hb_arraySetND( pRow, j + 1, randomValue );
         }
         hb_arraySet( pMatrix, i + 1, pRow );
         hb_itemRelease( pRow );
      }
      return pMatrix;
   }
   return hb_itemArrayNew(0);
}

// --- Workers de Operaciones Matriciales ---

static PHB_ITEM matrix_transpose( PHB_ITEM pMatrix )
{
   HB_SIZE nRows, nCols, i, j;
   PHB_ITEM pMatrixResult, pRow, pTransposedRow;
   if( !pMatrix || !HB_IS_ARRAY(pMatrix) || hb_arrayLen(pMatrix) == 0 ) return hb_itemArrayNew(0);

   nRows = hb_arrayLen( pMatrix );
   nCols = hb_arrayLen( hb_arrayGetItemPtr( pMatrix, 1 ) );
   pMatrixResult = hb_itemArrayNew( nCols );
   for( i = 0; i < nCols; i++ )
   {
      pTransposedRow = hb_itemArrayNew( nRows );
      for( j = 0; j < nRows; j++ )
      {
         pRow = hb_arrayGetItemPtr( pMatrix, j + 1 );
         hb_arraySetND( pTransposedRow, j + 1, hb_arrayGetND( pRow, i + 1 ) );
      }
      hb_arraySet( pMatrixResult, i + 1, pTransposedRow );
      hb_itemRelease( pTransposedRow );
   }
   return pMatrixResult;
}

static PHB_ITEM matrix_multiply( PHB_ITEM pMatrix1, PHB_ITEM pMatrix2 )
{
   HB_SIZE rows1, cols1, rows2, cols2, i, j, k;
   PHB_ITEM pResult, pRowResult, pRowA;
   double sum;
   if( !pMatrix1 || !pMatrix2 ) return hb_itemArrayNew(0);

   rows1 = hb_arrayLen( pMatrix1 );
   cols1 = (rows1 > 0) ? hb_arrayLen( hb_arrayGetItemPtr(pMatrix1, 1) ) : 0;
   rows2 = hb_arrayLen( pMatrix2 );
   cols2 = (rows2 > 0) ? hb_arrayLen( hb_arrayGetItemPtr(pMatrix2, 1) ) : 0;
   if (cols1 != rows2 || rows1 == 0 || rows2 == 0) return hb_itemArrayNew(0);

   pResult = hb_itemArrayNew( rows1 );
   for( i = 0; i < rows1; i++ )
   {
      pRowResult = hb_itemArrayNew( cols2 );
      pRowA = hb_arrayGetItemPtr( pMatrix1, i + 1 );
      for( j = 0; j < cols2; j++ )
      {
         sum = 0.0;
         for( k = 0; k < cols1; k++ )
         {
            sum += hb_arrayGetND( pRowA, k + 1 ) * hb_arrayGetND( hb_arrayGetItemPtr( pMatrix2, k + 1 ), j + 1 );
         }
         hb_arraySetND( pRowResult, j + 1, sum );
      }
      hb_arraySet( pResult, i + 1, pRowResult );
      hb_itemRelease( pRowResult );
   }
   return pResult;
}

static PHB_ITEM matrix_add_or_sub( PHB_ITEM pMatrix1, PHB_ITEM pMatrix2, int mode )
{
   HB_SIZE nRows, nCols, i, j;
   PHB_ITEM pMatrixResult, pRow1, pRow2, pRowResult;
   if( !pMatrix1 || !pMatrix2 || hb_arrayLen(pMatrix1) != hb_arrayLen(pMatrix2) ) return hb_itemArrayNew(0);
   nRows = hb_arrayLen(pMatrix1);
   if (nRows == 0) return hb_itemArrayNew(0);
   nCols = hb_arrayLen(hb_arrayGetItemPtr(pMatrix1, 1));

   pMatrixResult = hb_itemArrayNew( nRows );
   for( i = 0; i < nRows; i++ )
   {
      pRow1 = hb_arrayGetItemPtr( pMatrix1, i + 1 );
      pRow2 = hb_arrayGetItemPtr( pMatrix2, i + 1 );
      pRowResult = hb_itemArrayNew( nCols );
      for( j = 0; j < nCols; j++ )
      {
         double val1 = hb_arrayGetND(pRow1, j+1);
         double val2 = hb_arrayGetND(pRow2, j+1);
         hb_arraySetND( pRowResult, j + 1, (mode == 1) ? (val1 + val2) : (val1 - val2) );
      }
      hb_arraySet( pMatrixResult, i + 1, pRowResult );
      hb_itemRelease( pRowResult );
   }
   return pMatrixResult;
}
static PHB_ITEM matrix_add( PHB_ITEM pMatrix1, PHB_ITEM pMatrix2 ) { return matrix_add_or_sub(pMatrix1, pMatrix2, 1); }
static PHB_ITEM matrix_sub( PHB_ITEM pMatrix1, PHB_ITEM pMatrix2 ) { return matrix_add_or_sub(pMatrix1, pMatrix2, -1); }

static PHB_ITEM matrix_mul_scalar( PHB_ITEM pMatrix, double scalar )
{
   HB_SIZE nRows, nCols, i, j;
   PHB_ITEM pResult, pRow, pNewRow;
   
   if( !pMatrix || !HB_IS_ARRAY(pMatrix) || hb_arrayLen(pMatrix) == 0 ) return hb_itemArrayNew(0);
   
   nRows = hb_arrayLen(pMatrix);
   nCols = hb_arrayLen(hb_arrayGetItemPtr(pMatrix, 1));
   pResult = hb_itemArrayNew(nRows);
   
   for( i = 0; i < nRows; i++ )
   {
      pRow = hb_arrayGetItemPtr(pMatrix, i + 1);
      pNewRow = hb_itemArrayNew(nCols);
      for( j = 0; j < nCols; j++ )
      {
         hb_arraySetND(pNewRow, j + 1, hb_arrayGetND(pRow, j + 1) * scalar);
      }
      hb_arraySet(pResult, i + 1, pNewRow);
      hb_itemRelease(pNewRow);
   }
   return pResult;
}

static PHB_ITEM matrix_sum_axis0( PHB_ITEM pMatrix )
{
    HB_SIZE nRows, nCols, i, j;
    PHB_ITEM pResultRow, pRow, pResultMatrix;
    
    if (!pMatrix || hb_arrayLen(pMatrix) == 0) return hb_itemArrayNew(0);

    nRows = hb_arrayLen(pMatrix);
    nCols = hb_arrayLen(hb_arrayGetItemPtr(pMatrix, 1));
    pResultRow = hb_itemArrayNew(nCols);
    for (j = 0; j < nCols; j++) { hb_arraySetND(pResultRow, j + 1, 0.0); }

    for (i = 0; i < nRows; i++) {
        pRow = hb_arrayGetItemPtr(pMatrix, i + 1);
        for (j = 0; j < nCols; j++) {
            hb_arraySetND(pResultRow, j + 1, hb_arrayGetND(pResultRow, j + 1) + hb_arrayGetND(pRow, j + 1));
        }
    }
    // Para que sea una matriz 1xN
    pResultMatrix = hb_itemArrayNew(1);
    hb_arraySet(pResultMatrix, 1, pResultRow);
    hb_itemRelease(pResultRow);
    return pResultMatrix;
}

// --- Workers de Funciones de Activación ---

/*
* static PHB_ITEM relu_forward( PHB_ITEM pMatrix )
* ------------------------------------------------
* WORKER para la función de activación ReLU.
* VERSIÓN CORREGIDA Y ROBUSTA.
*/
static PHB_ITEM relu_forward( PHB_ITEM pMatrix )
{
   HB_SIZE rows, cols, i, j;
   PHB_ITEM pResult, pRow, pRowResult;

   // ====> ¡AQUÍ ESTÁ LA CORRECCIÓN! <====
   // Validar que la matriz no sea NULL, que sea un array y que no esté vacía.
   if( !pMatrix || !HB_IS_ARRAY(pMatrix) || hb_arrayLen(pMatrix) == 0 )
   {
      // Si la entrada no es válida, devolver un array vacío y no continuar.
      return hb_itemArrayNew(0);
   }

   // A partir de aquí, el código puede asumir que la matriz es válida.
   rows = hb_arrayLen(pMatrix);
   cols = hb_arrayLen(hb_arrayGetItemPtr(pMatrix, 1));
   pResult = matrix_zero( rows, cols );

   for( i = 0; i < rows; i++ )
   {
      pRow = hb_arrayGetItemPtr( pMatrix, i + 1 );
      pRowResult = hb_arrayGetItemPtr( pResult, i + 1 );
      for( j = 0; j < cols; j++ )
      {
         double val = hb_arrayGetND( pRow, j + 1 );
         hb_arraySetND( pRowResult, j + 1, val > 0 ? val : 0 );
      }
   }
   return pResult;
}
/*
* static PHB_ITEM softmax_forward( PHB_ITEM pValues )
* ---------------------------------------------------
* WORKER para la función Softmax.
* Transforma una matriz de puntuaciones (logits) en una matriz de probabilidades.
* Cada fila de la matriz de salida sumará 1.0.
*
* Parámetros:
* pValues: Una matriz (array de arrays) donde cada fila es un vector de puntuaciones.
*
* Retorna:
* Una nueva matriz con las probabilidades calculadas, o un array vacío si hay error.
*/
/*
static PHB_ITEM softmax_forward( PHB_ITEM pValues )
{
   HB_SIZE nRows, nCols, i, j;
   PHB_ITEM pResult, pRow, pRowResult;
   double sumExp, maxValue;
   double *expValues;

   // --- Validación de la entrada ---
   if( !pValues || !HB_IS_ARRAY(pValues) || hb_arrayLen(pValues) == 0 )
   {
      return hb_itemArrayNew(0);
   }
   nRows = hb_arrayLen( pValues );
   pRow = hb_arrayGetItemPtr(pValues, 1);
   if( !HB_IS_ARRAY(pRow) )
   {
       return hb_itemArrayNew(0);
   }
   nCols = hb_arrayLen(pRow);

   // --- Crear la matriz de resultado ---
   pResult = hb_itemArrayNew( nRows );

   // --- Bucle principal: procesar cada fila de forma independiente ---
   for( i = 0; i < nRows; i++ )
   {
      pRow = hb_arrayGetItemPtr( pValues, i + 1 );
      pRowResult = hb_itemArrayNew( nCols );
      sumExp = 0.0;
      maxValue = -DBL_MAX; // Inicializar con el valor doble más pequeño posible

      // --- 1. Encontrar el valor máximo en la fila (para estabilidad numérica) ---
      for( j = 0; j < nCols; j++ )
      {
         double val;
         val = hb_arrayGetND( pRow, j + 1 );
         if (val > maxValue)
         {
            maxValue = val;
         }
      }

      // --- 2. Calcular los exponentes (e^(x - max)) y su suma ---
      // Se utiliza un array temporal en C para mayor eficiencia.
      expValues = (double *) hb_xalloc( nCols * sizeof(double) );
      if( expValues == NULL )
      {
          // Manejar error de asignación de memoria
          hb_itemRelease(pResult);
          hb_itemRelease(pRowResult);
          return hb_itemArrayNew(0);
      }

      for( j = 0; j < nCols; j++ )
      {
         // Restar maxValue previene desbordamientos (overflow)
         double expValue;
         expValue = exp( hb_arrayGetND( pRow, j + 1 ) - maxValue );
         expValues[j] = expValue;
         sumExp += expValue;
      }

      // --- 3. Normalizar para obtener las probabilidades ---
      if (sumExp > 0)
      {
        for( j = 0; j < nCols; j++ )
        {
           hb_arraySetND( pRowResult, j + 1, expValues[j] / sumExp );
        }
      }
      else
      {
         // Si la suma es 0, se puede asignar una probabilidad uniforme o dejar en 0
         for( j = 0; j < nCols; j++ ) { hb_arraySetND( pRowResult, j + 1, 0.0 ); }
      }

      // Liberar memoria temporal y asignar la fila de resultado
      hb_xfree( expValues );
      hb_arraySet( pResult, i + 1, pRowResult );
      hb_itemRelease( pRowResult );
   }

   return pResult; // La función que llama debe liberar este resultado
}
*/

/*
* static PHB_ITEM matrix_add_broadcast( PHB_ITEM pMatrix, PHB_ITEM pVector )
* -------------------------------------------------------------------------
* WORKER para la suma con broadcasting. Suma un vector fila a cada fila de una matriz.
*
* Parámetros:
* pMatrix: La matriz principal (dimensiones M x N).
* pVector: Un vector representado como una matriz de 1 x N.
*
* Retorna:
* Una nueva matriz de M x N con el resultado, o un array vacío si hay error.
*/
static PHB_ITEM matrix_add_broadcast( PHB_ITEM pMatrix, PHB_ITEM pVector )
{
   HB_SIZE nRows, nCols, nBiasCols, i, j;
   PHB_ITEM pResult, pRow, pNewRow, pBiasRow;

   // --- Validación de Parámetros ---
   if( !pMatrix || !pVector || !HB_IS_ARRAY(pMatrix) || !HB_IS_ARRAY(pVector) )
   {
      // hb_errRT_BASE( ... ); // Podrías generar un error aquí
      return hb_itemArrayNew(0);
   }

   // El vector de bias debe ser una matriz de 1 fila
   if( hb_arrayLen(pVector) != 1 )
   {
      return hb_itemArrayNew(0);
   }
    
   nRows = hb_arrayLen( pMatrix );
   if( nRows == 0 )
   {
      return hb_itemArrayNew(0);
   }
    
   // Las dimensiones de las columnas deben coincidir
   nCols = hb_arrayLen( hb_arrayGetItemPtr( pMatrix, 1 ) );
   pBiasRow = hb_arrayGetItemPtr( pVector, 1 );
   nBiasCols = hb_arrayLen( pBiasRow );

   if( nCols != nBiasCols )
   {
      return hb_itemArrayNew(0);
   }

   // --- Operación de Broadcasting ---
   pResult = hb_itemArrayNew( nRows );
   for( i = 0; i < nRows; i++ )
   {
      pRow = hb_arrayGetItemPtr( pMatrix, i + 1 );
      pNewRow = hb_itemArrayNew( nCols );
      for( j = 0; j < nCols; j++ )
      {
         double matrixVal = hb_arrayGetND( pRow, j + 1 );
         double biasVal = hb_arrayGetND( pBiasRow, j + 1 );
         hb_arraySetND( pNewRow, j + 1, matrixVal + biasVal );
      }
      hb_arraySet( pResult, i + 1, pNewRow );
      hb_itemRelease( pNewRow );
   }

   return pResult; // La función que llama debe liberar este resultado
}

// ========================================================================
// SECCIÓN 3: IMPLEMENTACIÓN DE LOS WRAPPERS (API PARA HARBOUR)
// ========================================================================

HB_FUNC( HB_SEEDRAND ) { srand( ( unsigned int ) time( NULL ) ); }
HB_FUNC( HB_MATRIXCLONE ) { hb_itemReturnRelease( matrix_clone( hb_param(1, HB_IT_ARRAY) ) ); }
HB_FUNC( HB_MATRIXZERO ) { hb_itemReturnRelease( matrix_zero( hb_parns(1), hb_parns(2) ) ); }
HB_FUNC( HB_MATRIXRANDOM ) { hb_itemReturnRelease( matrix_random( hb_parns(1), hb_parns(2) ) ); }
HB_FUNC( HB_MATRIXTRANSPOSE ) { hb_itemReturnRelease( matrix_transpose( hb_param(1, HB_IT_ARRAY) ) ); }
HB_FUNC( HB_MATRIXMULTIPLY ) { hb_itemReturnRelease( matrix_multiply( hb_param(1, HB_IT_ARRAY), hb_param(2, HB_IT_ARRAY) ) ); }
HB_FUNC( HB_MATRIXADD ) { hb_itemReturnRelease( matrix_add( hb_param(1, HB_IT_ARRAY), hb_param(2, HB_IT_ARRAY) ) ); }
HB_FUNC( HB_MATRIXSUB ) { hb_itemReturnRelease( matrix_sub( hb_param(1, HB_IT_ARRAY), hb_param(2, HB_IT_ARRAY) ) ); }
HB_FUNC( HB_MATRIXMULSCALAR ) { hb_itemReturnRelease( matrix_mul_scalar( hb_param(1, HB_IT_ARRAY), hb_parnd(2) ) ); }
HB_FUNC( HB_MATRIXSUMAXIS0 ) { hb_itemReturnRelease( matrix_sum_axis0( hb_param(1, HB_IT_ARRAY) ) ); }
HB_FUNC( HB_RELU ) { hb_itemReturnRelease( relu_forward( hb_param(1, HB_IT_ARRAY) ) ); }
// ... (Más wrappers para el resto de funciones) ...


// ========================================================================
// SECCIÓN 4: IMPLEMENTACIÓN DE BACKPROPAGATION (API PARA HARBOUR)
// ========================================================================

HB_FUNC( HB_CROSSENTROPYLOSS )
{
    PHB_ITEM pPredictions = hb_param( 1, HB_IT_ARRAY );
    PHB_ITEM pTargets     = hb_param( 2, HB_IT_ARRAY );
    HB_SIZE nRows, nCols, i, j;
    double loss = 0.0;
    double *probs;

    if( !pPredictions || !pTargets || !HB_IS_ARRAY(pPredictions) || !HB_IS_ARRAY(pTargets) )
    {
        hb_retnd( 0.0 );
        return;
    }

    nRows = hb_arrayLen( pPredictions );
    if( nRows != hb_arrayLen( pTargets ) || nRows == 0 )
    {
        hb_retnd( 0.0 );
        return;
    }

    nCols = hb_arrayLen( hb_arrayGetItemPtr( pPredictions, 1 ) );
    if( nCols != hb_arrayLen( hb_arrayGetItemPtr( pTargets, 1 ) ) || nCols == 0 )
    {
        hb_retnd( 0.0 );
        return;
    }

    probs = (double*) hb_xalloc( nCols * sizeof(double) );

    // Compute cross-entropy loss with inline softmax
    for( i = 0; i < nRows; i++ )
    {
        PHB_ITEM pRowPred = hb_arrayGetItemPtr( pPredictions, i + 1 );
        PHB_ITEM pRowTargets = hb_arrayGetItemPtr( pTargets, i + 1 );
        double maxValue = -1e10;
        double sumExp = 0.0;

        // Find max for numerical stability
        for( j = 0; j < nCols; j++ )
        {
            double val = hb_arrayGetND( pRowPred, j + 1 );
            if( val > maxValue ) maxValue = val;
        }

        // Compute exp and sum
        for( j = 0; j < nCols; j++ )
        {
            double val = hb_arrayGetND( pRowPred, j + 1 );
            probs[j] = exp( val - maxValue );
            sumExp += probs[j];
        }

        // Normalize to probs
        for( j = 0; j < nCols; j++ )
        {
            probs[j] /= sumExp;
        }

        // Compute loss
        for( j = 0; j < nCols; j++ )
        {
            double target = hb_arrayGetND( pRowTargets, j + 1 );
            if( target > 0.0 )
            {
                loss -= target * log( probs[j] + 1e-10 ); // Add epsilon to avoid log(0)
            }
        }
    }

    hb_xfree( probs );
    hb_retnd( loss );
}


HB_FUNC( HB_RELU_BACKWARD )
{
   PHB_ITEM pDNextLayer = hb_param( 1, HB_IT_ARRAY );
   PHB_ITEM pForwardInput = hb_param( 2, HB_IT_ARRAY );
   HB_SIZE nRows = hb_arrayLen( pDNextLayer );
   HB_SIZE nCols = hb_arrayLen( hb_arrayGetItemPtr( pDNextLayer, 1 ) );
   PHB_ITEM pDInput = matrix_zero( nRows, nCols );
   HB_SIZE i, j;

   for( i = 0; i < nRows; i++ )
   {
      PHB_ITEM pRowD = hb_arrayGetItemPtr( pDNextLayer, i + 1 );
      PHB_ITEM pRowF = hb_arrayGetItemPtr( pForwardInput, i + 1 );
      PHB_ITEM pRowDInput = hb_arrayGetItemPtr( pDInput, i + 1 );
      for( j = 0; j < nCols; j++ )
      {
         if( hb_arrayGetND( pRowF, j + 1 ) > 0 )
         {
            hb_arraySetND( pRowDInput, j + 1, hb_arrayGetND( pRowD, j + 1 ) );
         }
      }
   }
   hb_itemReturnRelease( pDInput );
}

HB_FUNC( HB_MATRIXMULTIPLY_BACKWARD )
{
   PHB_ITEM pDC = hb_param( 1, HB_IT_ARRAY );
   PHB_ITEM pA  = hb_param( 2, HB_IT_ARRAY );
   PHB_ITEM pB  = hb_param( 3, HB_IT_ARRAY );
   PHB_ITEM pResultArray, pDA, pDB, pAT, pBT;

   if( !pDC || !pA || !pB ) { hb_ret(); return; }

   pBT = matrix_transpose(pB);
   pDA = matrix_multiply( pDC, pBT );
   hb_itemRelease( pBT );

   pAT = matrix_transpose(pA);
   pDB = matrix_multiply( pAT, pDC );
   hb_itemRelease( pAT );

   pResultArray = hb_itemArrayNew( 2 );
   hb_arraySet( pResultArray, 1, pDA );
   hb_arraySet( pResultArray, 2, pDB );
   hb_itemRelease( pDA );
   hb_itemRelease( pDB );
   hb_itemReturnRelease( pResultArray );
}


HB_FUNC( HB_MATRIXADDBROADCAST_BACKWARD )
{
   PHB_ITEM pDOutput = hb_param( 1, HB_IT_ARRAY );
   PHB_ITEM pDX, pDb, pResultArray;
   if( !pDOutput ) { hb_ret(); return; }

   pDX = matrix_clone( pDOutput );
   pDb = matrix_sum_axis0( pDOutput );

   pResultArray = hb_itemArrayNew( 2 );
   hb_arraySet( pResultArray, 1, pDX );
   hb_arraySet( pResultArray, 2, pDb );
   hb_itemRelease( pDX );
   hb_itemRelease( pDb );
   hb_itemReturnRelease( pResultArray );
}

// ... Las demás funciones de backpropagation como LAYERNORM_BACKWARD seguirían aquí ...
// (Omitidas por la extrema longitud, pero seguirían la estructura corregida)

// Las funciones originales que no llaman a otras funciones HB_... no necesitan refactorización
// Por ejemplo, HB_SGD_UPDATE, HB_ADAMUPDATE, HB_SOFTMAXBACKWARD, etc.
// ...

/*
* HB_FUNC( HB_ADAMUPDATE )
* ------------------------
* Actualiza los pesos (W) in-situ usando el optimizador Adam.
* Escrito en estricto estilo C89 (ANSI C).
*/
HB_FUNC( HB_ADAMUPDATE )
{
   // --- SECCIÓN ÚNICA DE DECLARACIONES (Estilo C89) ---
   // Punteros a Items de Harbour
   PHB_ITEM pW, pDW, pM, pV;
   PHB_ITEM pRowW, pRowDW, pRowM, pRowV;

   // Tipos de tamaño de Harbour y contadores de bucle
   HB_SIZE nRows, nCols, i, j;

   // Variables numéricas para el algoritmo
   int t;
   double lr, beta1, beta2, epsilon;
   double w, dw, m, v, m_hat, v_hat;

   // --- OBTENCIÓN DE PARÁMETROS ---
   pW  = hb_param(1, HB_IT_ARRAY);
   pDW = hb_param(2, HB_IT_ARRAY);
   pM  = hb_param(3, HB_IT_ARRAY);
   pV  = hb_param(4, HB_IT_ARRAY);
   t   = hb_parni(5);
   lr  = hb_parnd(6);
   // Parámetros opcionales con valores por defecto
   beta1   = HB_ISNIL(7) ? 0.9 : hb_parnd(7);
   beta2   = HB_ISNIL(8) ? 0.999 : hb_parnd(8);
   epsilon = HB_ISNIL(9) ? 0.00000001 : hb_parnd(9);

   // --- VALIDACIÓN DE PARÁMETROS ---
   if( !pW || !pDW || !pM || !pV || t <= 0 )
   {
      hb_errRT_BASE(EG_ARG, 3012, "Invalid parameters for HB_ADAM_UPDATE.", HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS);
      return;
   }
   // ... (Aquí irían más validaciones de dimensiones) ...

   // --- LÓGICA DEL ALGORITMO ---
   nRows = hb_arrayLen(pW);
   nCols = hb_arrayLen(hb_arrayGetItemPtr(pW, 1));

   for( i = 0; i < nRows; i++ )
   {
      pRowW = hb_arrayGetItemPtr(pW, i + 1);
      pRowDW = hb_arrayGetItemPtr(pDW, i + 1);
      pRowM = hb_arrayGetItemPtr(pM, i + 1);
      pRowV = hb_arrayGetItemPtr(pV, i + 1);

      for( j = 0; j < nCols; j++ )
      {
         // Obtener valores (las variables ya están declaradas arriba)
         w  = hb_arrayGetND(pRowW, j + 1);
         dw = hb_arrayGetND(pRowDW, j + 1);
         m  = hb_arrayGetND(pRowM, j + 1);
         v  = hb_arrayGetND(pRowV, j + 1);

         // Paso 1: Actualizar primer momento
         m = beta1 * m + (1.0 - beta1) * dw;
         hb_arraySetND(pRowM, j + 1, m);

         // Paso 2: Actualizar segundo momento
         v = beta2 * v + (1.0 - beta2) * (dw * dw);
         hb_arraySetND(pRowV, j + 1, v);

         // Paso 3: Corregir sesgo
         m_hat = m / (1.0 - pow(beta1, t));
         v_hat = v / (1.0 - pow(beta2, t));

         // Paso 4: Actualizar peso
         w = w - lr * m_hat / (sqrt(v_hat) + epsilon);
         hb_arraySetND(pRowW, j + 1, w);
      }
   }

   // --- RETORNO ---
   hb_itemReturn(pW);
}

HB_FUNC( HB_MATRIXFILL )
{
   PHB_ITEM pMatrix = hb_param(1, HB_IT_ARRAY);
   double value = hb_parnd(2);
   HB_SIZE nRows, i, j, nCols;
   PHB_ITEM pRow;

   if( pMatrix && HB_IS_NUMERIC(hb_param(2, HB_IT_NUMERIC)) )
   {
      nRows = hb_arrayLen(pMatrix);

      for( i = 0; i < nRows; i++ )
      {
         pRow = hb_arrayGetItemPtr(pMatrix, i + 1);
         nCols = hb_arrayLen(pRow);

         for( j = 0; j < nCols; j++ )
         {
            hb_arraySetND(pRow, j + 1, value);
         }
      }
      hb_itemReturn(pMatrix);
   }
   else
   {
      hb_errRT_BASE(EG_ARG, 3012, "Invalid parameters for HB_MATRIXFILL", HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS);
   }
}

HB_FUNC( HB_MATRIXDIVSCALAR )
{
   PHB_ITEM pMatrix = hb_param(1, HB_IT_ARRAY);
   double scalar = hb_parnd(2);
   HB_SIZE nRows, i, j, nCols;
   PHB_ITEM pResult, pRow, pNewRow;

   if( !pMatrix )
   {
      hb_errRT_BASE( EG_ARG, 3012, "Invalid matrix parameter.", HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS );
      return;
   }
   if( scalar == 0.0 )
   {
      hb_errRT_BASE( EG_ARG, 3012, "Division by zero.", HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS );
      return;
   }
   nRows = hb_arrayLen(pMatrix);
   pResult = hb_itemArrayNew(nRows);
   for( i = 0; i < nRows; i++ )
   {
      pRow = hb_arrayGetItemPtr(pMatrix, i + 1);
      nCols = hb_arrayLen(pRow);
      pNewRow = hb_itemArrayNew(nCols);
      for( j = 0; j < nCols; j++ )
      {
         hb_arraySetND(pNewRow, j + 1, hb_arrayGetND(pRow, j + 1) / scalar);
      }
      hb_arraySet(pResult, i + 1, pNewRow);
      hb_itemRelease(pNewRow);
   }
   hb_itemReturnRelease(pResult);
}

HB_FUNC( HB_SOFTMAX )
{
   PHB_ITEM pValues = hb_param( 1, HB_IT_ARRAY );
   HB_SIZE nRows, nCols, i, j;
   PHB_ITEM pResult, pRow, pRowResult;
   double sumExp, maxValue;

   if( !pValues )
   {
      hb_errRT_BASE( EG_ARG, 3012, "Invalid parameter.", HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS );
      return;
   }
   nRows = hb_arrayLen( pValues );
   if( nRows == 0 ) {
      hb_itemReturn( hb_itemArrayNew(0) );
      return;
   }
   nCols = hb_arrayLen( hb_arrayGetItemPtr( pValues, 1 ) );
   pResult = hb_itemArrayNew( nRows );

   for( i = 0; i < nRows; i++ )
   {
      pRow = hb_arrayGetItemPtr( pValues, i + 1 );
      pRowResult = hb_itemArrayNew( nCols );
      sumExp = 0.0;
      
      // Numerically stable softmax: find max value in row first
      if (nCols > 0) {
         maxValue = hb_arrayGetND( pRow, 1 );
         for( j = 1; j < nCols; j++ ) {
            double val = hb_arrayGetND( pRow, j + 1 );
            if (val > maxValue) maxValue = val;
         }
      } else {
          maxValue = 0.0;
      }
      
      // Calculate exponents and sum
      for( j = 0; j < nCols; j++ )
      {
         double expValue = exp( hb_arrayGetND( pRow, j + 1 ) - maxValue );
         hb_arraySetND( pRowResult, j + 1, expValue );
         sumExp += expValue;
      }

      // Normalize (avoid division by zero if sum is zero)
      if (sumExp > 0) {
        for( j = 0; j < nCols; j++ )
        {
           hb_arraySetND( pRowResult, j + 1, hb_arrayGetND( pRowResult, j + 1 ) / sumExp );
        }
      }
      hb_arraySet( pResult, i + 1, pRowResult );
      hb_itemRelease( pRowResult );
   }
   hb_itemReturnRelease( pResult );
}

HB_FUNC( HB_LAYERNORM )
{
   // Par├ímetros: 1. Matriz de entrada (X), 2. Matriz Gamma (╬│), 3. Matriz Beta (╬▓), 4. Epsilon (╬Á)
   PHB_ITEM pMatrix = hb_param( 1, HB_IT_ARRAY );
   PHB_ITEM pGamma  = hb_param( 2, HB_IT_ARRAY ); // Vector de escala (como matriz 1xN)
   PHB_ITEM pBeta   = hb_param( 3, HB_IT_ARRAY ); // Vector de desplazamiento (como matriz 1xN)
   double epsilon   = HB_ISNIL(4) ? 1e-5 : hb_parnd(4); // Epsilon con valor por defecto

   HB_SIZE nRows, nCols, i, j;
   PHB_ITEM pResult, pRow, pRowResult, pGammaRow, pBetaRow;
   double sum, mean, variance, val, inv_std_dev;

   // 1. --- Validaci├│n de Par├ímetros ---
   if( !pMatrix || !pGamma || !pBeta )
   {
      hb_errRT_BASE( EG_ARG, 3012, "Invalid parameters: NIL matrix passed to HB_LAYERNORM.", HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS );
      return;
   }

   nRows = hb_arrayLen( pMatrix );
   if( nRows == 0 )
   {
      hb_itemReturn( hb_itemArrayNew(0) ); // Devolver matriz vac├¡a si la entrada est├í vac├¡a
      return;
   }
   nCols = hb_arrayLen( hb_arrayGetItemPtr( pMatrix, 1 ) );

   // Gamma y Beta deben ser vectores (matrices de 1 fila) con el mismo n├║mero de columnas que la entrada
   if( hb_arrayLen(pGamma) != 1 || hb_arrayLen(pBeta) != 1 ||
       hb_arrayLen(hb_arrayGetItemPtr(pGamma, 1)) != nCols ||
       hb_arrayLen(hb_arrayGetItemPtr(pBeta, 1)) != nCols )
   {
      hb_errRT_BASE( EG_ARG, 3012, "Gamma and Beta must be 1xN matrices with N matching input columns.", HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS );
      return;
   }

   // 2. --- Preparaci├│n ---
   pResult = hb_itemArrayNew( nRows );
   pGammaRow = hb_arrayGetItemPtr(pGamma, 1);
   pBetaRow  = hb_arrayGetItemPtr(pBeta, 1);

   // 3. --- Bucle principal: Iterar sobre cada fila (cada token) ---
   for( i = 0; i < nRows; i++ )
   {
      pRow = hb_arrayGetItemPtr( pMatrix, i + 1 );

      // --- Paso A: Calcular la media (╬╝) de la fila actual ---
      sum = 0.0;
      for( j = 0; j < nCols; j++ )
      {
         sum += hb_arrayGetND( pRow, j + 1 );
      }
      mean = sum / nCols;

      // --- Paso B: Calcular la varianza (¤â^2) de la fila actual ---
      sum = 0.0;
      for( j = 0; j < nCols; j++ )
      {
         val = hb_arrayGetND( pRow, j + 1 ) - mean;
         sum += val * val;
      }
      variance = sum / nCols;

      // --- Paso C: Normalizar, escalar (╬│) y desplazar (╬▓) en un solo paso ---
      pRowResult = hb_itemArrayNew( nCols );
      inv_std_dev = 1.0 / sqrt( variance + epsilon ); // Optimizaci├│n: calcular la inversa una vez por fila

      for( j = 0; j < nCols; j++ )
      {
         double normalized, scaled, shifted;
         // Aplicar la f├│rmula: y = ((x - ╬╝) / sqrt(¤â^2 + ╬Á)) * ╬│ + ╬▓
         val = hb_arrayGetND( pRow, j + 1 );
         normalized = (val - mean) * inv_std_dev;
         scaled     = normalized * hb_arrayGetND( pGammaRow, j + 1 );
         shifted    = scaled + hb_arrayGetND( pBetaRow, j + 1 );

         hb_arraySetND( pRowResult, j + 1, shifted );
      }

      // A├▒adir la fila procesada a la matriz de resultado
      hb_arraySet( pResult, i + 1, pRowResult );
      hb_itemRelease( pRowResult );
   }

   // 4. --- Devolver la matriz resultante ---
   hb_itemReturnRelease( pResult );
}

/*
* HB_FUNC( HB_MATRIXADDBROADCAST )
* ---------------------------------
* WRAPPER para exponer la función matrix_add_broadcast a Harbour.
*
* Parámetros de Harbour:
* 1. Matriz (M x N)
* 2. Vector (1 x N)
*/
HB_FUNC( HB_MATRIXADDBROADCAST )
{
   // Obtener parámetros de Harbour
   PHB_ITEM pMatrix = hb_param( 1, HB_IT_ARRAY );
   PHB_ITEM pVector = hb_param( 2, HB_IT_ARRAY );

   // Llamar al worker para hacer el trabajo y devolver el resultado
   hb_itemReturnRelease( matrix_add_broadcast( pMatrix, pVector ) );
}

/*
* HB_FUNC( HB_LAYERNORM_BACKWARD )
* ---------------------------------
* Calcula los gradientes para la capa de Layer Normalization.
*
* La fórmula de forward es:
* x_hat = (x - mean) / sqrt(variance + epsilon)
* y = gamma * x_hat + beta
*
* Esta función calcula dL/dx, dL/dgamma y dL/dbeta usando dL/dy.
*
* Parámetros de Harbour:
* 1. dY (mDOutput): Gradiente de la capa siguiente (dL/dy).
* 2. X (mInput): La entrada original a la capa LayerNorm en el forward pass.
* 3. Gamma (mGamma): El vector de pesos gamma usado en el forward pass.
* 4. Epsilon (nEpsilon): El pequeño valor para evitar división por cero.
*
* Retorna:
* Un array de 3 elementos: { dX, dGamma, dBeta }
*/
HB_FUNC( HB_LAYERNORM_BACKWARD )
{
    // --- Variable declarations at the beginning (C89 compliance) ---
    PHB_ITEM pDY, pX, pGamma;
    PHB_ITEM pDX, pDGamma, pDBeta, pResultArray;
    PHB_ITEM pGammaRow, pDGammaRow, pDBetaRow;
    PHB_ITEM pXRow, pDYRow, pDXRow;
    HB_SIZE nRows, nCols, i, j, k;
    double epsilon, sum, mean, variance, inv_std_dev, val, x_hat, x_hat_k, dL_dy_k_gamma_k, sum_term1, sum_term2, x_hat_j, dL_dy_j_gamma_j, dx;
    
    // --- 1. Obtener Parámetros de Harbour ---
    pDY      = hb_param( 1, HB_IT_ARRAY ); // Gradiente de la salida
    pX       = hb_param( 2, HB_IT_ARRAY ); // Entrada original
    pGamma   = hb_param( 3, HB_IT_ARRAY ); // Pesos Gamma
    epsilon  = HB_ISNIL(4) ? 1e-5 : hb_parnd(4);

    // --- 2. Validación y Obtención de Dimensiones ---
    if( !pDY || !pX || !pGamma ) { hb_ret(); return; }
    nRows = hb_arrayLen(pX);
    if( nRows == 0 ) { hb_ret(); return; }
    nCols = hb_arrayLen(hb_arrayGetItemPtr(pX, 1));
    if( nCols == 0 ) { hb_ret(); return; }

    // --- 3. Inicializar Matrices de Gradientes ---
    pDX      = matrix_zero(nRows, nCols);
    pDGamma  = matrix_zero(1, nCols);
    pDBeta   = matrix_zero(1, nCols);
    pResultArray = hb_itemArrayNew(3);

    pGammaRow  = hb_arrayGetItemPtr(pGamma, 1);
    pDGammaRow = hb_arrayGetItemPtr(pDGamma, 1);
    pDBetaRow  = hb_arrayGetItemPtr(pDBeta, 1);

    // --- 4. Calcular dGamma y dBeta ---
    // Estos gradientes son la suma a través de todo el lote (todas las filas).
    for (i = 0; i < nRows; i++)
    {
        pXRow = hb_arrayGetItemPtr(pX, i + 1);
        pDYRow = hb_arrayGetItemPtr(pDY, i + 1);

        // Recalcular media y desviación estándar inversa para esta fila
        // NOTA: Para máxima eficiencia, estos valores se deberían "cachear"
        // durante el forward pass y pasar a esta función.
        sum = 0.0;
        for (j = 0; j < nCols; j++) { sum += hb_arrayGetND(pXRow, j + 1); }
        mean = sum / nCols;
        sum = 0.0;
        for (j = 0; j < nCols; j++) { val = hb_arrayGetND(pXRow, j + 1) - mean; sum += val * val; }
        variance = sum / nCols;
        inv_std_dev = 1.0 / sqrt(variance + epsilon);

        for (j = 0; j < nCols; j++)
        {
            x_hat = (hb_arrayGetND(pXRow, j + 1) - mean) * inv_std_dev;
            // Acumular gradiente para gamma: dL/dgamma = dL/dy * x_hat
            hb_arraySetND(pDGammaRow, j + 1, hb_arrayGetND(pDGammaRow, j + 1) + hb_arrayGetND(pDYRow, j + 1) * x_hat);
            // Acumular gradiente para beta: dL/dbeta = dL/dy
            hb_arraySetND(pDBetaRow, j + 1, hb_arrayGetND(pDBetaRow, j + 1) + hb_arrayGetND(pDYRow, j + 1));
        }
    }

    // --- 5. Calcular dX ---
    // Este es el paso más complejo, ya que el gradiente de cada x_i depende de todos los demás.
    for (i = 0; i < nRows; i++)
    {
        pXRow = hb_arrayGetItemPtr(pX, i + 1);
        pDYRow = hb_arrayGetItemPtr(pDY, i + 1);
        pDXRow = hb_arrayGetItemPtr(pDX, i + 1);
        
        // Recalcular de nuevo media y desviación para esta fila
        sum = 0.0;
        for (j = 0; j < nCols; j++) { sum += hb_arrayGetND(pXRow, j + 1); }
        mean = sum / nCols;
        sum = 0.0;
        for (j = 0; j < nCols; j++) { val = hb_arrayGetND(pXRow, j + 1) - mean; sum += val * val; }
        variance = sum / nCols;
        inv_std_dev = 1.0 / sqrt(variance + epsilon);

        // Términos intermedios para la fórmula de dX
        sum_term1 = 0.0;
        sum_term2 = 0.0;
        for (k = 0; k < nCols; k++)
        {
            x_hat_k = (hb_arrayGetND(pXRow, k + 1) - mean) * inv_std_dev;
            dL_dy_k_gamma_k = hb_arrayGetND(pDYRow, k + 1) * hb_arrayGetND(pGammaRow, k + 1);
            sum_term1 += dL_dy_k_gamma_k;
            sum_term2 += dL_dy_k_gamma_k * x_hat_k;
        }
        
        // Aplicar la fórmula final para cada elemento de la fila
        for (j = 0; j < nCols; j++)
        {
            x_hat_j = (hb_arrayGetND(pXRow, j + 1) - mean) * inv_std_dev;
            dL_dy_j_gamma_j = hb_arrayGetND(pDYRow, j + 1) * hb_arrayGetND(pGammaRow, j + 1);

            dx = (double)nCols * dL_dy_j_gamma_j;
            dx -= sum_term1;
            dx -= x_hat_j * sum_term2;
            dx *= inv_std_dev / (double)nCols;
            
            hb_arraySetND(pDXRow, j + 1, dx);
        }
    }

    // --- 6. Empaquetar y Devolver Resultados ---
    hb_arraySet(pResultArray, 1, pDX);
    hb_arraySet(pResultArray, 2, pDGamma);
    hb_arraySet(pResultArray, 3, pDBeta);

    hb_itemRelease(pDX);
    hb_itemRelease(pDGamma);
    hb_itemRelease(pDBeta);

    hb_itemReturnRelease(pResultArray);
}

/*
* HB_FUNC( HB_SOFTMAXBACKWARD )
* -----------------------------
* Calcula el gradiente a través de una capa Softmax (backpropagation).
*
* Parámetros de Harbour:
* 1. pDProbs: El gradiente de la capa siguiente (dL/dProbs).
* 2. pProbs: La salida de la capa Softmax del forward pass (las probabilidades).
*
* Retorna:
* El gradiente con respecto a la entrada de la capa Softmax (dL/dScores).
*/
HB_FUNC( HB_SOFTMAXBACKWARD )
{
   // --- Variable declarations (C89) ---
   PHB_ITEM pDProbs, pProbs, pDScores;
   PHB_ITEM pDProbsRow, pProbsRow, pDScoresRow;
   HB_SIZE nRows, nCols, i, j;
   double row_sum, prob_j, dprob_j, dscore_j;

   // --- 1. Obtener Parámetros y Validar ---
   pDProbs = hb_param(1, HB_IT_ARRAY); // Gradiente de las probabilidades
   pProbs  = hb_param(2, HB_IT_ARRAY);  // Salida de probabilidades del forward pass

   if (!pDProbs || !pProbs || !HB_IS_ARRAY(pDProbs) || !HB_IS_ARRAY(pProbs) ||
       hb_arrayLen(pDProbs) != hb_arrayLen(pProbs)) {
      hb_errRT_BASE(EG_ARG, 3012, "Invalid parameters or mismatched dimensions.", HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS);
      return;
   }
    
   nRows = hb_arrayLen(pProbs);
   if (nRows == 0) {
      hb_itemReturn(hb_itemArrayNew(0));
      return;
   }
   nCols = hb_arrayLen(hb_arrayGetItemPtr(pProbs, 1));

   // --- 2. Preparar Matriz de Resultado ---
   pDScores = hb_itemArrayNew(nRows);

   // --- 3. Bucle Principal: Procesar cada fila del lote ---
   for (i = 0; i < nRows; i++)
   {
      pDProbsRow = hb_arrayGetItemPtr(pDProbs, i + 1);
      pProbsRow = hb_arrayGetItemPtr(pProbs, i + 1);
      pDScoresRow = hb_itemArrayNew(nCols);
      row_sum = 0.0;

      // --- Paso A: Calcular el producto punto: row_sum = dProbs · Probs ---
      // Esto calcula una suma ponderada del gradiente entrante.
      for (j = 0; j < nCols; j++)
      {
         row_sum += hb_arrayGetND(pDProbsRow, j + 1) * hb_arrayGetND(pProbsRow, j + 1);
      }

      // --- Paso B: Aplicar la fórmula para obtener el gradiente de las puntuaciones ---
      // dScores = Probs * (dProbs - row_sum)
      for (j = 0; j < nCols; j++)
      {
         prob_j = hb_arrayGetND(pProbsRow, j + 1);
         dprob_j = hb_arrayGetND(pDProbsRow, j + 1);
         dscore_j = prob_j * (dprob_j - row_sum);
         hb_arraySetND(pDScoresRow, j + 1, dscore_j);
      }
        
      hb_arraySet(pDScores, i + 1, pDScoresRow);
      hb_itemRelease(pDScoresRow);
   }

   // --- 4. Devolver el Gradiente Calculado ---
   hb_itemReturnRelease(pDScores);
}

/*
* HB_FUNC( HB_SGD_UPDATE )
* ------------------------
* Actualiza una matriz de pesos (W) in-situ usando el Descenso de Gradiente Estocástico.
*
* La fórmula es: W_nuevo = W_viejo - tasa_de_aprendizaje * dW
* donde dW es el gradiente de la pérdida con respecto a W.
*
* Parámetros de Harbour:
* 1. pW: La matriz de pesos a actualizar (se modificará directamente).
* 2. pDW: La matriz de gradientes calculada durante la retropropagación.
* 3. lr: La tasa de aprendizaje (learning rate), un valor numérico.
*
* Retorna:
* Una referencia a la matriz de pesos actualizada (pW).
*/
HB_FUNC( HB_SGD_UPDATE )
{
   // --- Variable declarations at the beginning (C89 compliance) ---
   PHB_ITEM pW, pDW;
   PHB_ITEM pRowW, pRowDW;
   HB_SIZE nRowsW, nRowsDW, nColsW, nColsDW, i, j;
   double lr, current_weight, gradient, updated_weight;

   // --- 1. Obtener Parámetros de Harbour ---
   pW  = hb_param(1, HB_IT_ARRAY);   // Matriz de Pesos (Weights)
   pDW = hb_param(2, HB_IT_ARRAY);   // Matriz de Gradientes (dWeights)
   lr  = hb_parnd(3);                // Tasa de Aprendizaje (Learning Rate)

   // --- 2. Validación de Parámetros ---
   if( !pW || !pDW || !HB_IS_NUMERIC(hb_param(3, HB_IT_ANY)) )
   {
      hb_errRT_BASE(EG_ARG, 3012, "Invalid parameters for HB_SGD_UPDATE. Expected (Array, Array, Numeric).", HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS);
      return;
   }

   nRowsW = hb_arrayLen(pW);
   nRowsDW = hb_arrayLen(pDW);
   if ( nRowsW != nRowsDW || nRowsW == 0 )
   {
      hb_errRT_BASE(EG_ARG, 3012, "Mismatched row dimensions or empty matrix.", HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS);
      return;
   }

   nColsW = hb_arrayLen(hb_arrayGetItemPtr(pW, 1));
   nColsDW = hb_arrayLen(hb_arrayGetItemPtr(pDW, 1));
   if ( nColsW != nColsDW )
   {
      hb_errRT_BASE(EG_ARG, 3012, "Mismatched column dimensions.", HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS);
      return;
   }

   // --- 3. Bucle de Actualización In-Situ ---
   for( i = 0; i < nRowsW; i++ )
   {
      pRowW = hb_arrayGetItemPtr(pW, i + 1);
      pRowDW = hb_arrayGetItemPtr(pDW, i + 1);

      for( j = 0; j < nColsW; j++ )
      {
         // Obtener el peso y el gradiente actual
         current_weight = hb_arrayGetND(pRowW, j + 1);
         gradient = hb_arrayGetND(pRowDW, j + 1);

         // Aplicar la fórmula de SGD
         updated_weight = current_weight - lr * gradient;

         // Actualizar el valor directamente en la matriz de pesos original
         hb_arraySetND(pRowW, j + 1, updated_weight);
      }
   }

   // --- 4. Devolver una referencia a la matriz modificada ---
   hb_itemReturn(pW);
}

// --- Prototipos (añadir al inicio del archivo) ---
HB_FUNC( HB_MSE_LOSS );
HB_FUNC( HB_MSE_LOSS_BACKWARD );

// --- Implementaciones (añadir al final del archivo) ---

/*
* HB_FUNC( HB_MSE_LOSS )
* ----------------------
* Calcula el Error Cuadrático Medio entre dos matrices.
* Retorna un valor numérico escalar (el error).
* Formula: (1/N) * sum( (predicciones - objetivos)^2 )
*/
/*
* HB_FUNC( HB_MSE_LOSS )
* ----------------------
* Calcula el Error Cuadrático Medio. VERSIÓN CORREGIDA Y ROBUSTA.
*/
HB_FUNC( HB_MSE_LOSS )
{
   PHB_ITEM pPreds = hb_param(1, HB_IT_ARRAY);
   PHB_ITEM pTargets = hb_param(2, HB_IT_ARRAY);
   HB_SIZE nRows, nCols, i, j;
   double sum_sq_err = 0.0;

   // ====> ¡BLOQUE DE VALIDACIÓN AÑADIDO! <====
   if( !pPreds || !pTargets || !HB_IS_ARRAY(pPreds) || !HB_IS_ARRAY(pTargets) ||
       hb_arrayLen(pPreds) == 0 || hb_arrayLen(pTargets) == 0 )
   {
      hb_retnd( -1.0 ); // Devolver un error o un valor imposible
      return;
   }

   nRows = hb_arrayLen(pPreds);
   if ( nRows != hb_arrayLen(pTargets) )
   {
      hb_retnd( -1.0 ); // Las filas no coinciden
      return;
   }

   nCols = hb_arrayLen(hb_arrayGetItemPtr(pPreds, 1));
   if ( nCols != hb_arrayLen(hb_arrayGetItemPtr(pTargets, 1)) )
   {
      hb_retnd( -1.0 ); // Las columnas no coinciden
      return;
   }
   // --- Fin de la Validación ---


   for( i = 0; i < nRows; i++ )
   {
      PHB_ITEM pRowPred = hb_arrayGetItemPtr(pPreds, i + 1);
      PHB_ITEM pRowTarget = hb_arrayGetItemPtr(pTargets, i + 1);
      for( j = 0; j < nCols; j++ )
      {
         double diff = hb_arrayGetND(pRowPred, j + 1) - hb_arrayGetND(pRowTarget, j + 1);
         sum_sq_err += diff * diff;
      }
   }

   // Evitar división por cero si la matriz es válida pero no tiene elementos
   if ( nRows * nCols > 0)
   {
      hb_retnd( sum_sq_err / (nRows * nCols) );
   }
   else
   {
      hb_retnd( 0.0 );
   }
}

/*
* HB_FUNC( HB_MSE_LOSS_BACKWARD )
* -------------------------------
* Calcula el gradiente inicial para la retropropagación a partir del MSE.
* Retorna la matriz de gradiente.
* Formula: (2/N) * (predicciones - objetivos)
*/
HB_FUNC( HB_MSE_LOSS_BACKWARD )
{
   PHB_ITEM pPreds = hb_param(1, HB_IT_ARRAY);
   PHB_ITEM pTargets = hb_param(2, HB_IT_ARRAY);
   // ... (Validación de dimensiones omitida por brevedad) ...
   HB_SIZE nRows = hb_arrayLen(pPreds);
   HB_SIZE nCols = hb_arrayLen(hb_arrayGetItemPtr(pPreds, 1));

   // dLoss = Preds - Targets
   PHB_ITEM dLoss = matrix_sub(pPreds, pTargets); // Reutilizamos nuestro worker

   // dLoss = dLoss * (2/N)
   double scalar = 2.0 / (double)(nRows * nCols);
   HB_SIZE i, j;
   for( i = 0; i < nRows; i++ )
   {
      PHB_ITEM pRow = hb_arrayGetItemPtr(dLoss, i + 1);
      for( j = 0; j < nCols; j++ )
      {
         hb_arraySetND( pRow, j+1, hb_arrayGetND(pRow, j+1) * scalar );
      }
   }
   hb_itemReturn(dLoss); // matrix_sub ya crea una copia
}

/*
* HB_FUNC( HB_CROSSENTROPYLOSS_BACKWARD )
* ---------------------------------------
* Calcula el gradiente de la pérdida Cross-Entropy w.r.t. las probabilidades (o logits pre-softmax).
* Para CE + softmax combinado, dLoss/dProbs = (probs - targets).
* El escalado por tamaño de lote se debe hacer en el código de Harbour.
* Asume probs es matriz softmax (batch x vocab), targets one-hot (batch x vocab).
* Retorna matriz de gradientes (batch x vocab).
*/
HB_FUNC( HB_CROSSENTROPYLOSS_BACKWARD )
{
   // --- Variable declarations (C89) ---
   PHB_ITEM pProbs = hb_param(1, HB_IT_ARRAY);
   PHB_ITEM pTargets = hb_param(2, HB_IT_ARRAY);
   HB_SIZE nRows, nCols;
   PHB_ITEM pGrad;

   // --- Validación ---
   if( !pProbs || !pTargets || !HB_IS_ARRAY(pProbs) || !HB_IS_ARRAY(pTargets) ||
       hb_arrayLen(pProbs) == 0 || hb_arrayLen(pTargets) == 0 )
   {
      hb_ret();  // Empty array on error
      return;
   }

   nRows = hb_arrayLen(pProbs);
   if( nRows != hb_arrayLen(pTargets) )
   {
      hb_ret();
      return;
   }

   nCols = hb_arrayLen( hb_arrayGetItemPtr(pProbs, 1) );
   if( nCols != hb_arrayLen( hb_arrayGetItemPtr(pTargets, 1) ) )
   {
      hb_ret();
      return;
   }

   // --- Gradiente base: probs - targets ---
   pGrad = matrix_sub( pProbs, pTargets );  // Reutiliza worker

   // --- Retorno ---
   hb_itemReturnRelease( pGrad );
}

/*
* HB_FUNC( HB_DROPOUT )
* ---------------------
* Aplica dropout a una matriz. Durante el entrenamiento, pone a cero aleatoriamente
* algunos elementos con una probabilidad `rate` y escala los demás por `1 / (1 - rate)`.
*
* Parámetros de Harbour:
* 1. pMatrix: La matriz de entrada.
* 2. pRate: La probabilidad de dropout (un número).
* 3. pIsTraining: Un valor lógico que indica si está en modo de entrenamiento.
*
* Retorna:
* Un array de 2 elementos: { pResultMatrix, pMaskMatrix }
* pResultMatrix es la matriz con dropout aplicado.
* pMaskMatrix es la máscara utilizada (necesaria para el backward pass).
* Si no está en entrenamiento, devuelve la matriz original y una máscara de unos.
*/
HB_FUNC( HB_DROPOUT )
{
    PHB_ITEM pMatrix = hb_param( 1, HB_IT_ARRAY );
    double   rate    = hb_parnd( 2 );
    int      isTraining = hb_parl( 3 );
    HB_SIZE  nRows, nCols, i, j;
    PHB_ITEM pResult, pMask, pResultRow, pMaskRow, pRow;
    double   scale;
    PHB_ITEM pReturnArray;

    if( !pMatrix ) { hb_ret(); return; }

    nRows = hb_arrayLen( pMatrix );
    if ( nRows == 0 ) { hb_ret(); return; }
    nCols = hb_arrayLen( hb_arrayGetItemPtr( pMatrix, 1 ) );

    pResult = matrix_zero( nRows, nCols );
    pMask   = matrix_zero( nRows, nCols );

    if( !isTraining || rate == 0.0 )
    {
        // Si no está en entrenamiento o la tasa es 0, no hacer nada.
        // Solo devolver la matriz original y una máscara de unos.
        hb_itemRelease( pResult );
        pResult = hb_itemClone( pMatrix );
        for( i = 0; i < nRows; i++ )
        {
           pMaskRow = hb_arrayGetItemPtr(pMask, i + 1);
           for( j = 0; j < nCols; j++ ) { hb_arraySetND( pMaskRow, j + 1, 1.0 ); }
        }
    }
    else
    {
        scale = 1.0 / ( 1.0 - rate );
        for( i = 0; i < nRows; i++ )
        {
            pRow       = hb_arrayGetItemPtr( pMatrix, i + 1 );
            pResultRow = hb_arrayGetItemPtr( pResult, i + 1 );
            pMaskRow   = hb_arrayGetItemPtr( pMask, i + 1 );
            for( j = 0; j < nCols; j++ )
            {
                if( ( (double)rand() / RAND_MAX ) < rate )
                {
                    hb_arraySetND( pResultRow, j + 1, 0.0 );
                    hb_arraySetND( pMaskRow, j + 1, 0.0 );
                }
                else
                {
                    hb_arraySetND( pResultRow, j + 1, hb_arrayGetND( pRow, j + 1 ) * scale );
                    hb_arraySetND( pMaskRow, j + 1, scale );
                }
            }
        }
    }

    // Devolver un array con la matriz resultado y la máscara
    pReturnArray = hb_itemArrayNew( 2 );
    hb_arraySet( pReturnArray, 1, pResult );
    hb_arraySet( pReturnArray, 2, pMask );
    hb_itemRelease( pResult );
    hb_itemRelease( pMask );
    hb_itemReturnRelease( pReturnArray );
}


/*
* static PHB_ITEM matrix_elem_mult( PHB_ITEM pMatrix1, PHB_ITEM pMatrix2 )
* ----------------------------------------------------------------------
* WORKER para la multiplicación elemento a elemento.
*/
static PHB_ITEM matrix_elem_mult( PHB_ITEM pMatrix1, PHB_ITEM pMatrix2 )
{
   HB_SIZE nRows, nCols, i, j;
   PHB_ITEM pResult, pRow1, pRow2, pNewRow;

   if( !pMatrix1 || !pMatrix2 || hb_arrayLen(pMatrix1) != hb_arrayLen(pMatrix2) ) return hb_itemArrayNew(0);
   nRows = hb_arrayLen(pMatrix1);
   if (nRows == 0) return hb_itemArrayNew(0);
   nCols = hb_arrayLen(hb_arrayGetItemPtr(pMatrix1, 1));

   pResult = hb_itemArrayNew( nRows );
   for( i = 0; i < nRows; i++ )
   {
      pRow1 = hb_arrayGetItemPtr( pMatrix1, i + 1 );
      pRow2 = hb_arrayGetItemPtr( pMatrix2, i + 1 );
      pNewRow = hb_itemArrayNew( nCols );
      for( j = 0; j < nCols; j++ )
      {
         hb_arraySetND( pNewRow, j + 1, hb_arrayGetND(pRow1, j+1) * hb_arrayGetND(pRow2, j+1) );
      }
      hb_arraySet( pResult, i + 1, pNewRow );
      hb_itemRelease( pNewRow );
   }
   return pResult;
}

/*
* HB_FUNC( HB_DROPOUT_BACKWARD )
* ------------------------------
* Calcula el gradiente a través de una capa de dropout.
* Simplemente aplica la máscara al gradiente de la capa siguiente.
*
* Parámetros de Harbour:
* 1. pDOutput: El gradiente de la capa siguiente.
* 2. pMask: La máscara generada durante el forward pass de dropout.
*
* Retorna:
* El gradiente con respecto a la entrada de la capa de dropout (dOutput * mask).
*/
HB_FUNC( HB_DROPOUT_BACKWARD )
{
    PHB_ITEM pDOutput = hb_param( 1, HB_IT_ARRAY );
    PHB_ITEM pMask    = hb_param( 2, HB_IT_ARRAY );

    // El backward pass es simplemente una multiplicación elemento a elemento
    hb_itemReturnRelease( matrix_elem_mult( pDOutput, pMask ) );
}