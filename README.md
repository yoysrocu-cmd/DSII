Data Science II - Machine Learning para la Ciencia de Datos
Comisión #95405 Coderhouse 
---

# 🛡️ Detección de Fraude Financiero con Machine Learning

## 📋 Descripción del Proyecto
Este proyecto tiene como objetivo construir un modelo de Machine Learning capaz de identificar transacciones financieras fraudulentas. Se aborda un problema real de la industria financiera donde el fraude representa una pérdida económica significativa y un riesgo para la confianza de los clientes.

El proyecto ha sido desarrollado como parte del **Módulo 16: Cierre - Proyecto Final** del curso de Ciencia de Datos, aplicando las mejores prácticas de la industria: desde el *Data Wrangling* y el Análisis Exploratorio de Datos (EDA), hasta la optimización de hiperparámetros y la explicabilidad del modelo con SHAP.

---

## 🎯 Formulación del Problema

*   **Contexto de Negocio:** Una entidad financiera necesita identificar transacciones fraudulentas para evitar pérdidas económicas y proteger a sus clientes.
*   **Tipo de Tarea:** Clasificación Binaria.
*   **Variable Objetivo (Target):** `fraud_label` (1 = Fraude, 0 = Transacción normal).
*   **Métrica de Éxito:** 
    *   **AUC-ROC:** Para medir la capacidad de discriminación global del modelo.
    *   **F1-Score:** Debido al alto desbalance de clases (solo ~0.8% son fraudes), esta métrica es crucial para evaluar el rendimiento en la clase minoritaria.
    *   *Objetivo de Negocio:* Minimizar los Falsos Negativos (dejar pasar fraudes), ya que el costo de un fraude no detectado es mucho mayor que el de una falsa alarma.
*   **Dataset:** `fraud_1500.csv` (1500 registros, 25 columnas). Contiene información transaccional, temporal, de comportamiento del usuario y de riesgo del comercio.

---

## 📂 Contenido del Repositorio

*   `Entrega_II_Proy_Fraude.ipynb`: Notebook principal de Google Colab con todo el pipeline (EDA, Feature Engineering, Modelado, Optimización y SHAP).
*   `fraud_1500.csv`: Dataset utilizado para el entrenamiento y evaluación.
*   `README.md`: Este archivo de documentación.

---

## ⚙️ Tecnologías y Librerías Utilizadas

*   **Lenguaje:** Python 3.10+
*   **Manipulación de Datos:** Pandas, NumPy
*   **Visualización:** Matplotlib, Seaborn
*   **Machine Learning:** Scikit-Learn (Pipelines, Preprocesamiento, GridSearchCV), XGBoost
*   **Explicabilidad:** SHAP (SHapley Additive exPlanations)
*   **Entorno:** Google Colab

---

## 🚀 Guía de Reproducibilidad (Cómo ejecutar el proyecto)

Sigue estos pasos para reproducir los resultados obtenidos:

1.  **Clonar el repositorio o descargar los archivos:**
    ```bash
    git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
    ```
2.  **Abrir el Notebook:**
    *   Puedes abrir el archivo `Entrega_II_Proy_Fraude.ipynb` directamente en [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12Z5Jl2H3bDysw4NFV66rwlgUsXWXJWxu?usp=sharing) o en tu entorno local de Jupyter.
3.  **Cargar el Dataset:**
    *   Si estás en Colab, ejecuta la celda de carga de datos y sube el archivo `fraud_1500.csv` cuando se te solicite. *(El código ya está preparado para esto).*
    *   Si estás en local, asegúrate de que el archivo `.csv` esté en la misma carpeta que el notebook.
4.  **Ejecutar todas las celdas:**
    *   Ve al menú `Entorno de ejecución` > `Ejecutar todas` (o `Runtime` > `Run all`).
    *   El notebook instalará automáticamente las librerías necesarias si no las tienes (aunque Colab ya las incluye por defecto).

---

## 📊 Metodología y Pipeline del Proyecto

El proyecto sigue un flujo de trabajo estructurado y reproducible:

1.  **Data Wrangling:** Limpieza inicial, eliminación de identificadores únicos (`transaction_id`) y verificación de valores nulos.
2.  **Análisis Exploratorio de Datos (EDA):** 
    *   Análisis de la distribución del target (alto desbalance).
    *   Visualización de la distribución del monto de transacción respecto al fraude (Boxplots).
    *   Matriz de correlación de variables numéricas.
3.  **Ingeniería de Atributos:**
    *   Uso de `ColumnTransformer` para aplicar `StandardScaler` a variables numéricas y `OneHotEncoder` a variables categóricas.
4.  **Entrenamiento y Optimización:**
    *   Se implementaron **Pipelines** de Scikit-Learn para evitar la fuga de datos (*data leakage*).
    *   **Modelo Base:** Regresión Logística (con `class_weight='balanced'`).
    *   **Modelo Avanzado:** XGBoost (configurado con `scale_pos_weight` para manejar el desbalance).
    *   **Optimización:** Búsqueda de hiperparámetros con `GridSearchCV` (evaluando `n_estimators`, `max_depth` y `learning_rate`).
5.  **Evaluación y Explicabilidad:**
    *   Evaluación en el conjunto de Test (AUC-ROC y F1-Score).
    *   Matriz de Confusión para visualizar Falsos Positivos y Negativos.
    *   Uso de **SHAP** para interpretar qué variables influyen más en la predicción de fraude (importancia global y local).

---

## 📈 Resultados y Conclusiones

*(Nota: Reemplaza estos valores con los que obtengas al ejecutar tu notebook).*

*   **Rendimiento del Modelo:** El modelo XGBoost optimizado logró un **AUC-ROC de [0.95 y 099]** en el conjunto de test, demostrando una excelente capacidad para distinguir entre transacciones normales y fraudulentas.
*   **Manejo del Desbalance:** Gracias al uso de `scale_pos_weight` y la métrica F1-Score, el modelo logró detectar una proporción significativa de los fraudes reales, a pesar de que representan menos del 1% del total de los datos.
*   **Explicabilidad (SHAP):** El análisis SHAP reveló que las variables más determinantes para predecir un fraude son:
    1.  `merchant_risk_score` (Puntaje de riesgo del comercio)
    2.  `amount` (Monto de la transacción)
    3.  `ip_risk_score` (Puntaje de riesgo de la dirección IP)

*   💡**Implicación de Negocio:** 
     *   Un `merchant_risk_score` alto indica que el comercio tiene un historial previo de fraudes, por lo que el modelo aprende a desconfiar de él.
     *   Montos (`amount`) muy altos o atípicos (comparados con el promedio histórico del cliente) son un detonante claro de alerta.
     *   Un `ip_risk_score` elevado sugiere que la transacción se origina desde una ubicación o red sospechosa (ej. VPNs, proxies o países de alto riesgo).
     *   Esto permite al equipo de riesgo priorizar sus alertas en estas tres características, optimizando sus recursos.

---

## ✍️ Autor

*   **JOHANNA ROCHA CUEVAS**
*   [Mi Perfil de GitHub](https://github.com/yoysrocu-cmd)
*   Proyecto desarrollado para el Módulo 16 - Cierre: Proyecto Final.


*   Referencias
Financial Fraud: Labeled Transaction Data (2026)
[Financial Fraud Dataset en Kaggle](https://www.kaggle.com/datasets/mubashirsidiki/financial-fraud-labeled-transaction-data-2026?resource=download)
