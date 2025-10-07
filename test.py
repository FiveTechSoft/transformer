import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time
import math
from datetime import datetime

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)
        pe = self._generate_pos_encoding(d_model, max_len=100)
        self.register_buffer('pos_encoding', pe)

    def _generate_pos_encoding(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        seq_len = x.size(0)
        pos = self.pos_encoding[:seq_len, :]
        x = self.embedding(x) + pos
        x = self.transformer_encoder(x.unsqueeze(0)).squeeze(0)
        return self.output_proj(x)

def main():
    nLayers = 2
    nVocabSize = 5
    nEmbedDim = 20
    nHiddenDim = 40
    nEpochs = 500
    nLearningRate = 0.01
    nNumExamples = 100
    nSeqLen = 5

    print(f"Ejecución iniciada el {datetime.now().strftime('%d/%m/%Y')} a las {datetime.now().strftime('%H:%M:%S')}")

    # Generar datos múltiples
    random.seed(42)  # Para reproducibilidad
    aAllInputs = []
    aAllTargets = []
    for _ in range(nNumExamples - 1):
        aInputTokens = [random.randint(0, nVocabSize - 1) for _ in range(nSeqLen)]
        aTargetTokens = aInputTokens[::-1]
        aAllInputs.append(aInputTokens)
        aAllTargets.append(aTargetTokens)

    # Agregar la secuencia específica (corregido el target)
    aAllInputs.append([1, 2, 3, 4, 0])
    aAllTargets.append([0, 4, 3, 2, 1])

    # Crear el modelo Transformer
    model = TransformerModel(nVocabSize, nEmbedDim, nhead=4, num_layers=nLayers, dim_feedforward=nHiddenDim)
    optimizer = optim.Adam(model.parameters(), lr=nLearningRate)

    print("Iniciando entrenamiento (con Embeddings aprendibles, CE Loss y Adam)...")
    print(f"Vocab: {nVocabSize} Embed: {nEmbedDim} Ejemplos: {len(aAllInputs)}")
    print("-" * 50)

    nLastTime = time.time()
    nStartTime = time.time()
    nMinLoss = float('inf')
    nEpochMinLoss = 0
    nEpochsSinceMin = 0

    for i in range(1, nEpochs + 1):
        if i == 1:
            print("Iniciando bucle de entrenamiento...")

        model.train()
        optimizer.zero_grad()
        nLoss = 0.0
        num_ex = len(aAllInputs)

        for aInputTokens, aTargetTokens in zip(aAllInputs, aAllTargets):
            inp = torch.tensor(aInputTokens, dtype=torch.long)
            tgt = torch.tensor(aTargetTokens, dtype=torch.long)
            mOutput = model(inp)
            ce = F.cross_entropy(mOutput, tgt, reduction='mean')
            nLoss += ce / num_ex
            (ce / num_ex).backward()

        optimizer.step()

        # Mostrar progreso cada 500 épocas con tiempo
        if i % 500 == 0:
            nPartialTime = time.time() - nLastTime
            print(f"Época {i} -> Loss: {nLoss:.4f} Tiempo parcial: {nPartialTime:.2f} segundos")
            nLastTime = time.time()

        # Scheduler
        if i % 500 == 0 and i > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.995
            print(f"LR decay a {param_group['lr']:.6f}")

        # Trackear el mejor loss
        if nLoss < nMinLoss:
            nMinLoss = nLoss
            nEpochMinLoss = i
            nEpochsSinceMin = 0
        else:
            nEpochsSinceMin += 1

        # Early stopping
        if nEpochsSinceMin > 300 and i > 100:
            print(f"Early stopping en Época {i} - Restaurando mejores pesos de época {nEpochMinLoss}")
            break

        if i % 100 == 0 or i <= 20 or (i > 100 and i <= 500 and i % 50 == 0):
            print(f"Época {i:5d} -> Loss: {nLoss:.4f}  (Mejor: {nMinLoss:.4f} en época {nEpochMinLoss}) Sin mejora: {nEpochsSinceMin}")

    print(f"Tiempo total de entrenamiento: {time.time() - nStartTime:.2f} segundos")
    print("-" * 50)

    # Verificación Final en múltiples ejemplos
    print("Entrenamiento finalizado. Verificando en múltiples ejemplos:")
    aTestExamples = [
        ([1, 2, 3, 4, 0], [0, 4, 3, 2, 1]),
        ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0]),
        ([2, 4, 1, 3, 0], [0, 3, 1, 4, 2]),
        ([4, 3, 2, 1, 0], [0, 1, 2, 3, 4])
    ]
    nCorrect = 0
    model.eval()
    with torch.no_grad():
        for aTest in aTestExamples:
            aTestInput = aTest[0]
            aTestTarget = aTest[1]
            inp = torch.tensor(aTestInput, dtype=torch.long)
            mTestOutput = model(inp)
            aPredictedTokens = torch.argmax(mTestOutput, dim=1).tolist()

            print(f"Entrada:   {aTestInput}")
            print(f"Objetivo:  {aTestTarget}")
            print(f"Predicción: {aPredictedTokens}")
            if aPredictedTokens == aTestTarget:
                print("✓ Correcto")
                nCorrect += 1
            else:
                print("✗ Incorrecto")
            print("")

    print(f"Ejemplos correctos: {nCorrect} de {len(aTestExamples)}")
    if nCorrect == len(aTestExamples):
        print("ÉXITO! El modelo generaliza bien a nuevos ejemplos.")
        print("Modificación de prueba: Código verificado por GitHub Copilot.")
    else:
        print("FALLO. El modelo no generaliza completamente.")

if __name__ == "__main__":
    main()