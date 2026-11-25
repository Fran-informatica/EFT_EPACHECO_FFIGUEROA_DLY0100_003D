# Sistema multimodal de imágenes y texto — EFT DLY0100

Evaluación Final Transversal de **Deep Learning (DLY0100)**, sección 003D, Duoc UC, Escuela de Informática y Telecomunicaciones.

**Integrantes:** Esteban Pacheco Vega y Francisco Figueroa Donoso.
**Docente:** Guillermo Eugenio Pinto Fuentes.

El sistema recibe una imagen de una prenda y una reseña de texto, y predice una clase conjunta de 20 categorías (prenda × sentimiento, por ejemplo «Bolso · positiva»). Se construye en tres notebooks de PyTorch:

1. Una **red convolucional (CNN)** clasifica las 10 prendas de Fashion-MNIST.
2. **Redes recurrentes (RNN simple, LSTM y GRU) y un Transformer** clasifican el sentimiento de reseñas de IMDB.
3. Un **clasificador fusionado** combina la representación de cada modelo; se comparan la concatenación + MLP y la atención entre modalidades.

## Contenido del repositorio

| Archivo | Contenido |
|---|---|
| `EFT_Esteban_y_Francisco_003D_CNN_FashionMNIST_v1.ipynb` | Parte 1: CNN sobre Fashion-MNIST, experimentos, evaluación en test y exportación de `cnn_final.pt` |
| `EFT_Esteban_y_Francisco_003D_Texto_IMDB_v1.ipynb` | Parte 2: RNN simple, LSTM, GRU y Transformer sobre IMDB, comparación entre familias y exportación de `texto_final.pt` |
| `EFT_Esteban_y_Francisco_003D_Fusion_Multimodal_v1.ipynb` | Parte 3: fusión de ambas representaciones, comparación de técnicas, evaluación en test y demostración del sistema completo |
| `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte` | Fashion-MNIST en formato IDX: 60.000 imágenes de entrenamiento y 10.000 de prueba, de 28 × 28 píxeles |
| `IMDB Dataset.csv` | 50.000 reseñas de IMDB con su sentimiento (25.000 positivas y 25.000 negativas) |

## Cómo ejecutar

Los notebooks se ejecutan en **Google Colab con GPU T4** (*Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4*). Cada notebook lee los datos desde la carpeta de trabajo o, si no están, directamente desde este repositorio. No hace falta subir archivos a mano ni montar Google Drive.

| Paso | Notebook | Abrir en Colab | Archivos que produce |
|---|---|---|---|
| 1 | CNN | [Abrir](https://colab.research.google.com/github/Fran-informatica/EFT_EPACHECO_FFIGUEROA_DLY0100_003D/blob/main/EFT_Esteban_y_Francisco_003D_CNN_FashionMNIST_v1.ipynb) | `respaldo_eft_cnn.zip` (incluye `cnn_final.pt`) y `resultados_cnn_guardados.json` |
| 2 | Texto | [Abrir](https://colab.research.google.com/github/Fran-informatica/EFT_EPACHECO_FFIGUEROA_DLY0100_003D/blob/main/EFT_Esteban_y_Francisco_003D_Texto_IMDB_v1.ipynb) | `texto_final.pt`, `respaldo_eft_texto.zip` y `resultados_texto_guardados.json` |
| 3 | Fusión | [Abrir](https://colab.research.google.com/github/Fran-informatica/EFT_EPACHECO_FFIGUEROA_DLY0100_003D/blob/main/EFT_Esteban_y_Francisco_003D_Fusion_Multimodal_v1.ipynb) | `respaldo_eft_fusion.zip` y `resultados_fusion_guardados.json` |

**Entre un paso y otro:**

- **Descargar los archivos** que produce cada notebook antes de cerrar la sesión de Colab: quedan en el servidor y se pierden al cerrarla.
- **Antes del notebook de fusión,** subir `cnn_final.pt` (está dentro de `respaldo_eft_cnn.zip`) y `texto_final.pt` a la raíz del repositorio o a la carpeta de trabajo de Colab. La fusión carga ambos modelos con `weights_only=True`.
- **Para no repetir entrenamientos:**
  - Cada notebook descarga solo, al terminar cada celda que entrena, el archivo `resultados_*_guardados.json`. También descarga los `.pt` de las corridas que necesitan el modelo o sus curvas: `corrida_cnn_*.pt`, `corrida_texto_*.pt` o `corrida_fusion_*.pt`.
  - El JSON va en la raíz del repositorio, sin el número que agrega el navegador a las copias repetidas, por ejemplo `(1)`. Los `.pt` van en `corridas_cnn/`, `corridas_texto/` o `corridas_fusion/`, o en la raíz.
  - Cada notebook reutiliza las corridas cuya configuración, semilla, épocas y paciencia coinciden exactamente.
  - Los `.pt` de texto pesan unos 20 MB cada uno, y en total suman unos 300 MB entre los tres notebooks. Conviene subirlos con git en vez de la web, que limita a 25 MB por archivo.
- **Si una sesión se corta,** cada corrida nueva queda impresa en la salida de su celda como una línea `CORRIDA_GUARDADA {...}`, a partir de la cual se puede reconstruir el JSON.

## Metodología común

- **Particiones:**
  - Fashion-MNIST: 50.000 / 10.000 / 10.000 (entrenamiento, validación y test), con la validación estratificada sacada del entrenamiento oficial y el test oficial.
  - IMDB: las 418 reseñas duplicadas se eliminan antes de partir, y las 49.582 restantes se dividen en 34.708 / 7.437 / 7.437, estratificadas por sentimiento.
  - Fusión: cada imagen se empareja al azar con una reseña de su misma partición.
- **Métrica de decisión:** F1 macro en validación, en los tres notebooks. El test se usa una sola vez, con el modelo final.
- **Experimentos:**
  - Se cambia un hiperparámetro a la vez sobre la configuración vigente.
  - Antes de comparar, se mide el rango de error entre reentrenamientos con 5 semillas (42, 1, 2, 3 y 4). Una diferencia menor o igual a ese rango es un empate técnico.
  - Si la mejor variante queda en el borde del rango probado, el rango se amplía.
  - Al final se verifican las combinaciones con interacción conocida y se compara el modelo final con el base usando las mismas semillas.
- **Texto:**
  - Minúsculas, sin etiquetas `<br />` y conservando la puntuación.
  - Largo máximo de 320 tokens (conserva cerca del 80 % de los tokens), truncando desde el final para conservar el inicio de la reseña.
  - Vocabulario con las palabras que aparecen al menos 5 veces en entrenamiento y *embedding* de 128 valores.
- **Representaciones:** los modelos de imagen y de texto entregan a la fusión un vector de 128 valores cada uno (su última capa oculta) y se usan congelados.

## Fuentes de los datos

- **Fashion-MNIST:** Xiao, H., Rasul, K. y Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*. arXiv:1708.07747. Repositorio oficial: https://github.com/zalandoresearch/fashion-mnist (licencia MIT).
- **IMDB:** Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y. y Potts, C. (2011). *Learning Word Vectors for Sentiment Analysis*. ACL. Sitio del dataset: https://ai.stanford.edu/~amaas/data/sentiment/

## Herramientas

Google Colab (GPU NVIDIA T4), Visual Studio Code con la extensión de Google Colab, PyTorch, NumPy, Pandas, scikit-learn y Plotly. Como asistente de IA se usó Claude Opus 5.5 (Anthropic) para estructurar los notebooks y apoyar el código y la redacción; los resultados, las decisiones y las interpretaciones se revisaron con las salidas de los propios notebooks. El detalle está en el anexo de herramientas de cada notebook.
