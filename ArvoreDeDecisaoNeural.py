import numpy as np
import tensorflow as tf

class ArvoreDeDecisaoNeural:
    def __init__(self, modelo: tf.keras.Sequential):
        # extração dos pesos da rede para facilitar
        self.camadas = []
        for layer in modelo.layers:
            w, b = layer.get_weights()
            self.camadas.append((w, b))

        self.ultima_funcao_ativacao = modelo.layers[-1].activation 

    def prever(self, x0_original):
        """
        Implementação do Algoritmo 1 do artigo.
        """
        x0 = np.append(x0_original, 1.0)

        w_inicial, b_inicial = self.camadas[0]

        W_efetivo = np.vstack([w_inicial, b_inicial])
        
        n = len(self.camadas)
        # de 0 a n-2
        for i in range(n - 1):
            pre_ativacao = np.dot(x0, W_efetivo)

            # O vetor de inclinação "a" direto, representando nossa decisão 
            a = (pre_ativacao > 0).astype(float)

            W_proximo, b_proximo = self.camadas[i + 1]

            W_proximo_filtrado = W_proximo * a[:, np.newaxis]

            W_efetivo = np.dot(W_efetivo, W_proximo_filtrado)

            W_efetivo[-1, :] += b_proximo

        return self.ultima_funcao_ativacao(np.dot(x0, W_efetivo))