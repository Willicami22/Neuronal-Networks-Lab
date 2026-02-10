# Laboratorio de Redes Neuronales: Capas Convolucionales

<div align="center">

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.13+-D00000?style=for-the-badge&logo=keras&logoColor=white)

**Exploración experimental del poder de las capas convolucionales en la clasificación de imágenes**

</div>

---

##  Tabla de Contenidos

- [Resumen del Ejercicio](#-resumen-del-ejercicio)
- [Descripción del Problema](#-descripción-del-problema)
- [Descripción del Dataset](#-descripción-del-dataset)
- [Arquitecturas](#-arquitecturas)
- [Resultados Experimentales](#-resultados-experimentales)
- [Interpretación y Análisis](#-interpretación-y-análisis)
- [Comenzando](#-comenzando)
  - [Prerrequisitos](#prerrequisitos)
  - [Instalación](#instalación)
  - [Ejecución](#ejecución)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Despliegue](#-despliegue)
- [Construido Con](#-construido-con)
- [Autores](#-autores)

---

##  Resumen del Ejercicio

Este laboratorio explora de manera experimental **por qué las capas convolucionales son tan efectivas para la clasificación de imágenes**. A través de experimentos controlados, se compara:

1. **Baseline (MLP)**: Una red neuronal completamente conectada tradicional
2. **CNN Personalizada**: Una arquitectura convolucional diseñada desde cero
3. **Experimento Controlado**: Comparación del efecto del tamaño del kernel (3×3 vs 5×5)

### Objetivos de Aprendizaje

- Entender las ventajas de las convoluciones sobre las redes densas para datos visuales
- Analizar el impacto de decisiones arquitectónicas específicas
- Realizar experimentos controlados variando un solo hiperparámetro
- Interpretar resultados y trade-offs en el diseño de CNNs

---

##  Descripción del Problema

### Pregunta de Investigación

**¿Por qué las capas convolucionales funcionan mejor que las capas densas para clasificación de imágenes?**

Para responder esto, el laboratorio implementa:

1. **Diseño de arquitectura CNN** desde cero 
2. **Justificación de decisiones arquitectónicas** (número de filtros, tamaño de kernels, pooling, etc.)
3. **Experimentos controlados** que aíslan variables específicas
4. **Comparación cuantitativa** contra un baseline MLP
5. **Análisis interpretativo** de los resultados

### Hipótesis

Las convoluciones deberían superar al MLP porque:
- **Preservan la estructura espacial** de las imágenes
- Implementan **weight sharing** (reutilización de filtros)
- Construyen **características jerárquicas** (edges → texturas → partes → objetos)
- Imponen un **sesgo inductivo** hacia localidad y equivarianza translacional

---

##  Descripción del Dataset

### CIFAR-100

<div align="center">

| Característica | Valor |
|:--------------|------:|
| **Total de imágenes** | 60,000 |
| **Imágenes de entrenamiento** | 50,000 |
| **Imágenes de prueba** | 10,000 |
| **Dimensiones** | 32×32×3 (RGB) |
| **Número de clases** | 100 |
| **Distribución** | Balanceado |

</div>

### ¿Por qué CIFAR-100 es ideal para CNNs?

1. **Estructura espacial 2D**: Imágenes RGB de tamaño fijo (32×32×3) que explotan la localidad espacial
2. **Complejidad jerárquica**: 100 categorías requieren aprender patrones desde bordes básicos hasta objetos complejos
3. **Tamaño manejable**: Suficientemente pequeño para experimentación rápida, pero lo suficientemente complejo para evaluar capacidades
4. **Variabilidad visual**: Amplia gama de categorías (animales, vehículos, objetos, plantas, etc.)

### Análisis Exploratorio (EDA)

El notebook incluye:
- Visualización de muestras aleatorias por clase
- Distribución de clases
- Análisis de intensidad de píxeles
- Estadísticas de normalización

---

##  Arquitecturas

### 1. Baseline: Multi-Layer Perceptron (MLP)

**Arquitectura completamente conectada** que sirve como punto de comparación.

```
Input (32×32×3 = 3,072 features)
    ↓
  Flatten
    ↓
Dense(512) + ReLU + Dropout(0.3)
    ↓
Dense(256) + ReLU + Dropout(0.3)
    ↓
Dense(128) + ReLU + Dropout(0.2)
    ↓
Dense(100) + Softmax
    ↓
Output (100 clases)
```

**Características:**
- **Parámetros**: ~1.7M
- **Limitaciones**: 
  - Destruye la estructura espacial con `Flatten`
  - No explota localidad ni patrones compartidos
  - Alta capacidad pero pobre generalización en imágenes

---

### 2. CNN Personalizada (Kernel 3×3)

**Arquitectura convolucional** diseñada específicamente para este problema.

```
Input (32×32×3)
    ↓
Block 1:
  Conv2D(32 filters, 3×3) + ReLU + Same Padding
  Conv2D(32 filters, 3×3) + ReLU + Same Padding
  MaxPooling2D(2×2) 
  Dropout(0.25)
    ↓  [Feature maps: 16×16×32]
    
Block 2:
  Conv2D(64 filters, 3×3) + ReLU + Same Padding
  Conv2D(64 filters, 3×3) + ReLU + Same Padding
  MaxPooling2D(2×2)
  Dropout(0.25)
    ↓  [Feature maps: 8×8×64]
    
Block 3:
  Conv2D(128 filters, 3×3) + ReLU + Same Padding
  GlobalAveragePooling2D
    ↓  [Feature vector: 128]
    
Classifier:
  Dense(128) + ReLU + Dropout(0.3)
  Dense(100) + Softmax
    ↓
Output (100 clases)
```

**Decisiones de diseño:**

| Decisión | Justificación |
|:---------|:--------------|
| **Kernels 3×3** | Balance óptimo: pequeños y apilables, capturan contexto local sin exceso de parámetros |
| **Padding 'same'** | Preserva dimensiones espaciales, evita pérdida de información en bordes |
| **MaxPooling 2×2** | Reduce dimensionalidad espacial gradualmente, crea invarianza a pequeñas traslaciones |
| **Profundidad creciente** | 32→64→128 filtros captura jerarquía de features (simple→complejo) |
| **GlobalAveragePooling** | Reduce sobreajuste vs Flatten, hace red robusta a posición de objeto |
| **Dropout estratégico** | 0.25 después de convoluciones, 0.3 en capas densas para regularización |

**Parámetros**: ~169K (10× menos que MLP)

---

### 3. Experimento Controlado: Comparación de Tamaño de Kernel

**Variable independiente**: Tamaño de kernel (3×3 vs 5×5)  
**Variables controladas**: Todo lo demás (número de filtros, profundidad, dropout, optimizador, learning rate, etc.)

#### CNN con Kernel 5×5

Misma arquitectura que la CNN 3×3, **solo cambiando** el tamaño del kernel a 5×5.

**Parámetros**: ~416K (2.46× más que 3×3)

---

##  Resultados Experimentales

### Comparación General

<div align="center">

| Modelo | Parámetros | Train Acc | Val Acc | Train Loss | Val Loss |
|:-------|----------:|----------:|--------:|-----------:|---------:|
| **MLP (Baseline)** | 1,700,000+ | 16.95% | **16.26%** | 3.5454 | 3.6191 |
| **CNN 3×3** | 168,836 | — | **19.85%** | — | 3.2477 |
| **CNN 5×5** | 416,132 | — | **24.83%** | — | 3.0408 |

</div>

### Experimento Controlado: Tamaño de Kernel

<div align="center">

| Kernel | Parámetros | Best Val Acc | Best Val Loss | Test Acc | Test Loss |
|:------:|-----------:|-------------:|--------------:|---------:|----------:|
| **3×3** | 168,836 | 19.85% | 3.2477 | 5.28% | 4.1364 |
| **5×5** | 416,132 | **24.83%** | **3.0408** | **7.59%** | **3.9482** |

</div>

**Radio de parámetros (5×5 vs 3×3)**: 2.46×

### Observaciones Clave

#### 1. CNN vs MLP
-  Las CNNs **superan significativamente** al MLP baseline
-  Incluso con **10× menos parámetros**, la CNN 3×3 obtiene mejor accuracy
-  Confirma la hipótesis: las convoluciones explotan mejor la estructura espacial

#### 2. Tamaño de Kernel (3×3 vs 5×5)
-  **5×5 mejora el accuracy** (~5% en validación)
-  **Costo**: 2.46× más parámetros y entrenamiento más lento
-  **Trade-off**: Mayor contexto por capa vs eficiencia computacional

---

##  Interpretación y Análisis

### ¿Por qué las Convoluciones Funcionan Mejor?

#### 1. **Preservación de Estructura Espacial**
- El MLP aplasta la imagen (32×32×3) en un vector 1D de 3,072 elementos
- Las convoluciones mantienen la relación espacial (píxeles vecinos)
- Ejemplo: Un "borde vertical" es un patrón espacial que se pierde al aplanar

#### 2. **Weight Sharing (Reutilización de Filtros)**
- MLP: Cada conexión tiene un peso único (millones de parámetros)
- CNN: El mismo filtro 3×3 se aplica a toda la imagen (~27 parámetros por filtro)
- Resultado: Menos parámetros, mejor generalización

#### 3. **Invarianza Translacional**
- Un filtro detecta el mismo patrón (ej: "ojo de gato") sin importar dónde aparezca
- MLP necesitaría aprender "ojo en posición (5,10)", "ojo en posición (8,15)", etc.

#### 4. **Jerarquía de Features**
```
Capa 1 (32 filtros):  Bordes, texturas simples
       ↓
Capa 2 (64 filtros):  Patrones, combinaciones de bordes
       ↓
Capa 3 (128 filtros): Partes de objetos (ojos, ruedas, ventanas)
       ↓
Dense:                 Objetos completos (gato, carro, avión)
```

---

### Kernels 3×3 vs 5×5: Trade-offs

#### Ventajas de 5×5
-  **Mayor campo receptivo** por capa (25 píxeles vs 9)
-  Captura **contexto más amplio** en cada operación
-  **Mejor accuracy** en este experimento (+5%)

#### Desventajas de 5×5
-  **2.46× más parámetros** (mayor riesgo de overfitting)
-  **Entrenamiento más lento** (más operaciones por convolución)
-  **Mayor memoria** requerida

#### Filosofía de Diseño Moderno
**Preferencia por kernels pequeños (3×3)**:
- Apilar dos capas 3×3 = campo receptivo 5×5
- Apilar tres capas 3×3 = campo receptivo 7×7
- **Ventaja**: Más no-linealidades (ReLU) entre capas → mayor capacidad expresiva
- **Resultado**: Arquitecturas modernas (ResNet, VGG) usan casi exclusivamente 3×3

---

### ¿Cuándo NO usar Convoluciones?

Las convoluciones son **menos apropiadas** cuando:

1. **Datos sin estructura espacial**
   - Ejemplo: Datos tabulares (edad, ingreso, código postal)
   - Razón: No hay "vecindad" significativa entre features

2. **Orden de features es arbitrario**
   - Ejemplo: Características no ordenadas de un paciente
   - Razón: Weight sharing asume que patrones son relevantes en cualquier posición

3. **Interacciones globales dominan**
   - Ejemplo: Grafos completamente conectados
   - Razón: Convoluciones explotan localidad; si todo interactúa con todo, no hay ventaja

4. **Datos 1D sin periodicidad**
   - Excepción: Señales temporales o audio SÍ se benefician de convoluciones 1D

---

##  Comenzando

### Prerrequisitos

- **Python**: 3.9 o superior (probado con 3.12)
- **Sistema Operativo**: Windows, macOS o Linux
- **Memoria RAM**: Mínimo 8GB recomendado
- **IDE**: Jupyter Notebook, JupyterLab o VS Code con extensión de Jupyter

### Instalación

#### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/neuronal-networks-lab.git
cd neuronal-networks-lab
```

#### 2. Crear entorno virtual

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install numpy matplotlib tensorflow
```

**Versiones específicas (probadas):**
```bash
pip install numpy==2.4.2 matplotlib==3.10.8 tensorflow==2.20.0
```

### Ejecución

#### Opción 1: Jupyter Notebook

```bash
jupyter notebook Neuronal-network-lab.ipynb
```

#### Opción 2: VS Code

1. Abrir `Neuronal-network-lab.ipynb`
2. Seleccionar el kernel `.venv`
3. Ejecutar celdas secuencialmente

#### Orden de Ejecución

El notebook está estructurado para ejecutarse **de arriba a abajo**:

1. **Setup**: Instalación de paquetes
2. **Dataset Definition**: Carga de CIFAR-100
3. **EDA**: Análisis exploratorio
4. **Baseline MLP**: Entrenamiento y evaluación
5. **Custom CNN**: Entrenamiento y evaluación
6. **Controlled Experiment**: Comparación 3×3 vs 5×5

 **Tiempo de ejecución estimado**: 30-45 minutos (depende de hardware)


---

##  Despliegue

### AWS SageMaker (Planeado - No Implementado)

Se preparó un flujo de trabajo para despliegue escalable en SageMaker, pero **no pudo ejecutarse** debido a restricciones de permisos en AWS Academy Learner Lab.

#### Proceso de Despliegue Planeado

1. **Serialización del Modelo**
   - Exportar pesos, arquitectura y preprocesamiento
   - Empaquetar en formato compatible con SageMaker

2. **Configuración de Training Job**
   - Estimador personalizado de TensorFlow
   - Tipo de instancia: `ml.p3.2xlarge` (GPU)
   - Hiperparámetros: learning rate, batch size, epochs

3. **Creación de Endpoint**
   - Deployment del modelo entrenado
   - Auto-scaling configurado
   - Health checks y monitoreo

4. **Testing de Inferencia**
   ```python
   # Ejemplo de predicción
   input_data = {
       "instances": [imagen_normalizada.tolist()]
   }
   response = predictor.predict(input_data)
   # Output: {"predictions": [[0.02, 0.15, ..., 0.67]]}
   ```

#### Limitaciones de AWS Academy

 **Restricciones que impidieron el despliegue:**
- Permisos IAM insuficientes para crear training jobs
- Acceso a SageMaker bloqueado en cuentas educativas
- Restricciones en S3 para artifacts de modelos
- No se permite provisioning de instancias EC2 para endpoints
- Credenciales temporales con tiempo de expiración limitado

---

##  Construido Con

### Frameworks y Librerías

- **[TensorFlow](https://www.tensorflow.org/)** `2.20.0` - Framework de deep learning
- **[Keras](https://keras.io/)** `3.13.2` - API de alto nivel para redes neuronales
- **[NumPy](https://numpy.org/)** `2.4.2` - Operaciones numéricas y arrays
- **[Matplotlib](https://matplotlib.org/)** `3.10.8` - Visualización de datos

### Herramientas de Desarrollo

- **Python** `3.12.6`
- **Jupyter Notebook** - Entorno interactivo
- **Git** - Control de versiones

### Dataset

- **[CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)** - Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009

---

##  Autores

- **William Hernandez** 
  - Universidad: Universidad Escuela Colombiana de Ingenieria Julio Garavito
  

---


##  Licencia

Este proyecto es un ejercicio académico para fines educativos.

