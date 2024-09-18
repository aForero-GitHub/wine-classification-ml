# Evaluación de Modelos para la Clasificación de Vinos

En este documento, se presenta el proceso de evaluación y análisis de diferentes modelos de clasificación para el conjunto de datos de vinos. Los modelos que se probaron incluyen **Random Forest**, **Logistic Regression**, y **Decision Tree**. A continuación, se muestra un resumen del proceso seguido y los resultados obtenidos.

## Proceso de Evaluación

1. **Carga y Preprocesamiento de Datos**:
   - El conjunto de datos fue descargado desde el repositorio de UCI. Se utilizaron técnicas de preprocesamiento como el manejo de valores atípicos mediante la imputación de la mediana.
   - Las características del conjunto de datos incluyen 13 variables que describen las propiedades químicas y visuales del vino.
   - Se utilizó el método **IQR** para identificar valores atípicos y realizar una imputación adecuada.

2. **División de los Datos**:
   - El conjunto de datos se dividió en dos partes: 80% para entrenamiento y 20% para prueba, con una semilla de aleatoriedad para asegurar resultados reproducibles.

3. **Selección de Modelos**:
   - Se seleccionaron los siguientes modelos de clasificación:
     - **Random Forest**
     - **Logistic Regression**
     - **Decision Tree**
   - La selección de estos modelos se debió a su efectividad en problemas de clasificación multicategoría.

4. **Métricas de Evaluación**:
   - Se utilizaron las siguientes métricas para evaluar el desempeño de los modelos:
     - **Precisión (Accuracy)**
     - **Precisión ponderada (Weighted Precision)**
     - **Recall ponderado (Weighted Recall)**
     - **F1 Score**
   - En algunos casos, también se evaluó el **AUC-ROC** para medir el área bajo la curva de recepción.

5. **Uso de MLflow**:
   - Se utilizó **MLflow** para el seguimiento de experimentos, registro de métricas y trazabilidad de los modelos.
   - Cada uno de los experimentos se ejecutó en **Databricks**, donde se registraron automáticamente los resultados utilizando `mlflow.autolog()`.

## Resultados Obtenidos

### Random Forest

- **Precisión**: `0.9688`
- **Precision ponderada**: `0.9731`
- **Recall ponderado**: `0.9677`
- **F1-Score**: `0.9687`

![Resultados de Random Forest](experiments.png)

### Decision Tree

- **Precisión**: `0.9661`
- **Precision ponderada**: `0.9642`
- **Recall ponderado**: `0.9674`
- **F1-Score**: `0.9658`

![Métricas de Decision Tree](GrphDecisionTree.png)
![Detalles de Métricas](MetricsDecisionTree.png)

### Logistic Regression

- **Precisión**: `0.9731`
- **Precision ponderada**: `0.9687`
- **Recall ponderado**: `0.9712`
- **F1-Score**: `0.9721`

![Métricas de Logistic Regression](GrphLogisticRegression.png)
![Detalles de Métricas](MetricsLogisticRegression.png)

## Conclusiones

1. **Volatilidad en los Resultados**:
   - Durante las pruebas locales, el modelo **Random Forest** mostró un rendimiento superior con una precisión de `0.9688`. Sin embargo, al migrar las pruebas a **Databricks**, el modelo de **Logistic Regression** sobresalió con una precisión ligeramente mayor de `0.9731`.
   - Este comportamiento puede deberse a la volatilidad en la partición de los datos, las configuraciones específicas del entorno y la optimización de hiperparámetros.

2. **Elección del Mejor Modelo**:
   - Aunque **Logistic Regression** presentó el mejor rendimiento en **Databricks**, ambos modelos (Random Forest y Logistic Regression) mostraron métricas muy competitivas. Dependiendo del caso de uso, ambos modelos pueden considerarse efectivos para esta tarea.

3. **Integración de Herramientas**:
   - La integración de herramientas como **Spark**, **MLflow** y **Databricks** permite una trazabilidad completa de los experimentos, facilitando la comparación de modelos y optimización en entornos de producción.
   - **MLflow** se utilizó para el registro automático de todos los experimentos, lo que mejora significativamente la eficiencia y el monitoreo de los modelos en un flujo de trabajo de machine learning.

4. **Futuras Mejoras**:
   - A medida que se integren más datos o se ajuste el pipeline de entrenamiento, se pueden realizar optimizaciones adicionales, como el ajuste de hiperparámetros utilizando técnicas como **Grid Search** o **Random Search**.

---

Este archivo incluye las imágenes y los gráficos generados durante la evaluación de los modelos. Para que las imágenes se muestren correctamente, asegúrate de que las imágenes estén subidas en el mismo repositorio en la carpeta adecuada.
