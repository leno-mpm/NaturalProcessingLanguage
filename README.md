# 📊 Clasificación de Emociones en Tweets con ML y Deep Learning

Este proyecto implementa y compara **tres enfoques diferentes** de clasificación de emociones en tweets:  
1. **Regresión Logística (TF-IDF)**  
2. **Red LSTM (Embeddings + Secuencias)**  
3. **Fine-tuning de BERT Multilingual**  

El objetivo es identificar correctamente emociones como *alegría, tristeza, miedo, ira, sorpresa, disgusto, otros*, y analizar el desempeño de cada modelo en diferentes eventos.

---

## 🧪 Modelos Implementados

### 🔹 Modelo 1: Logistic Regression (Baseline)
- **Representación:** TF-IDF (1-2 ngramas, `max_features=5000`)
- **Algoritmo:** `LogisticRegression(max_iter=500)`
- **Métricas:** Accuracy, Precision, Recall, F1-score
- **Ventaja:** rápido de entrenar, sirve como referencia.

---

### 🔹 Modelo 2: LSTM
- **Representación:** Embeddings trainables + Secuencias
- **Arquitectura:**
  - `Embedding(10000, 128)`
  - `LSTM(128, dropout=0.2, recurrent_dropout=0.2)`
  - `Dense(num_classes, softmax)`
- **Entrenamiento:** 5 épocas, `batch_size=64`
- **Métricas:** igual que el modelo baseline.
- **Ventaja:** aprende dependencias secuenciales en el texto.

---

### 🔹 Modelo 3: BERT (Transformers)
- **Pre-entrenamiento:** `bert-base-multilingual-cased`
- **Fine-tuning:** con nuestro dataset
- **Representación:** contextual de palabras
- **Entrenamiento:** con GPU (Colab recomendado)
- **Ventaja:** mejor desempeño esperado en clasificación de texto.


---
## 🚀 Cómo Ejecutar

Corre el Notebook sin preocipaciones de **subir archivos**: la data se **lee directamente desde GitHub**.
