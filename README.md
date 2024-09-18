# [Video Explicativo Documentacion](https://drive.google.com/file/d/13bFVWcflROi9-mZ2d8bNmTSbcOkp7-fL/view?usp=sharing)  <-

# --ES--

# Clasificación de Vinos Utilizando Modelos de Machine Learning en Databricks

## Resumen del Proyecto
Este proyecto tiene como objetivo clasificar variedades de vino en función de sus propiedades químicas, utilizando diferentes modelos de machine learning. El conjunto de datos utilizado para este problema de clasificación proviene del [repositorio de UCI sobre vinos](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data). Los datos contienen 13 propiedades químicas de tres tipos de cultivos de vino, y el objetivo es predecir el tipo de vino en función de estas características.

El proyecto utiliza **Spark** y **Python** para el procesamiento de datos, el entrenamiento de modelos y la evaluación, ya que estas tecnologías son conocidas por su rapidez, flexibilidad y capacidad para manejar grandes volúmenes de datos. Se utilizó la plataforma Databricks Community Edition para gestionar los experimentos, asegurar la trazabilidad y monitorear el desempeño de los modelos con **MLflow**.

## Conjunto de Datos
- **Fuente**: [Conjunto de datos de Vinos en UCI](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data)
- **Descripción**: El conjunto de datos consta de 178 muestras de vinos cultivados en la misma región de Italia, pero derivados de tres diferentes cultivos. El conjunto de datos contiene 13 atributos químicos (como Alcohol, Ácido málico, Cenizas, etc.) y una etiqueta de clase que representa el tipo de vino (Clase 1, Clase 2 o Clase 3).
- **Atributos**:
  1. Alcohol
  2. Ácido málico
  3. Cenizas
  4. Alcalinidad de las cenizas
  5. Magnesio
  6. Fenoles totales
  7. Flavonoides
  8. Fenoles no flavonoides
  9. Proantocianinas
  10. Intensidad de color
  11. Matiz
  12. OD280/OD315 de vinos diluidos
  13. Prolina

## Objetivos
1. **Exploración de Datos**: Realizar un análisis exploratorio de datos (EDA) para entender el conjunto de datos, identificar relaciones entre variables y detectar valores atípicos o faltantes.
2. **Preprocesamiento de Datos**: Manejar valores faltantes, detectar y tratar valores atípicos utilizando el **método IQR**, y preparar el conjunto de datos para el entrenamiento de los modelos.
3. **Entrenamiento de Modelos**: Entrenar varios modelos de machine learning, incluyendo:
   - Regresión Logística
   - Árbol de Decisión
   - Bosque Aleatorio (Random Forest)
4. **Evaluación de Modelos**: Evaluar los modelos en base a métricas como la exactitud, precisión, recall, F1-score, y AUC-ROC (adaptado para clasificación multiclase).
5. **Selección del Mejor Modelo**: Identificar el modelo con mejor rendimiento y explicar los criterios para su elección.
6. **Reducción de Dimensionalidad**: Aplicar **Análisis de Componentes Principales (PCA)** para reducir la dimensionalidad y observar su impacto en el rendimiento del modelo.
7. **Ajuste de Hiperparámetros**: Utilizar validación cruzada y ajuste de hiperparámetros para optimizar el modelo de Random Forest.

## Herramientas y Librerías Utilizadas
- **Spark**: Para procesamiento de datos distribuidos y entrenamiento de modelos.
- **Python**: Para scripting y lógica personalizada.
- **Pandas**: Para manipulación de datos.
- **ydata_profiling**: Para generar un informe exploratorio inicial de los datos.
- **Matplotlib/Seaborn**: Para visualizaciones.
- **MLflow**: Para el seguimiento de experimentos y métricas de modelos.
- **Databricks Community Edition**: Para registrar experimentos, entrenamientos y métricas.

## Preprocesamiento de Datos
- **Detección de Valores Atípicos**: Se identificaron valores atípicos utilizando el **método IQR**. Estos valores atípicos fueron tratados para mejorar la robustez del modelo.
  - **Cálculo del IQR**: Para cada característica, se calculó el IQR como Q3 - Q1, y los puntos de datos que caían por debajo de Q1 - 1.5*IQR o por encima de Q3 + 1.5*IQR fueron considerados como atípicos.
  - **Imputación**: Los valores atípicos fueron imputados utilizando el valor mediano de la respectiva característica para mitigar el impacto de los valores extremos.

## Selección de Modelos
Se entrenaron tres modelos para resolver el problema de clasificación:

1. **Regresión Logística**: Un modelo simple que a menudo es efectivo para problemas de clasificación multiclase.
2. **Árbol de Decisión**: Un modelo capaz de capturar fronteras de decisión complejas y manejar relaciones no lineales.
3. **Random Forest**: Un modelo en conjunto robusto, conocido por su capacidad para reducir el sobreajuste y manejar datos de alta dimensionalidad.

**¿Por qué estos modelos?**
- **Regresión Logística** a menudo se utiliza como modelo base para problemas multiclase y ofrece una fácil interpretabilidad.
- **Árbol de Decisión** maneja bien la no linealidad y requiere menos preparación de datos.
- **Random Forest** agrega múltiples árboles de decisión, proporcionando alta precisión y robustez.

### Métricas Utilizadas para la Evaluación
Se utilizaron las siguientes métricas para evaluar el rendimiento de los modelos:
- **Exactitud (Accuracy)**: Medida de la corrección global del modelo.
- **Precisión, Recall, F1-score**: Para evaluar el rendimiento específico de cada clase.
- **AUC-ROC**: Adaptado para problemas multiclase utilizando la metodología de "uno contra todos".
- **Matriz de Confusión**: Visualizada para evaluar el rendimiento de las clases.

### Resultados del Mejor Modelo: Random Forest
Después de entrenar y comparar los tres modelos, **Random Forest** mostró el mejor rendimiento con una exactitud del **96.88%**. A continuación, se muestran los resultados para dos nuevas muestras:

| Características                                                            | Predicción | Probabilidad           |
|---------------------------------------------------------------------|------------|-----------------------|
| [13.72, 1.43, 2.5, 16.7, 108, 3.4, 3.67, 0.19, 2.04, 6.8, 0.89, 2.87, 1285] | 1.0        | [0.0, 1.0, 0.0, 0.0]  |
| [12.37, 0.94, 1.36, 10.6, 88, 1.98, 0.57, 0.28, 0.42, 1.95, 1.05, 1.82, 520]  | 2.0        | [0.0, 0.0, 0.75, 0.25]|

## Reducción de Dimensionalidad con PCA
Se aplicó **PCA** para reducir la dimensionalidad. Sin embargo, esto resultó en una ligera disminución en la exactitud al **90.62%**, lo cual es esperable ya que se pierde algo de varianza en el proceso de reducción dimensional.

## Ajuste de Hiperparámetros y Validación Cruzada
Finalmente, se aplicó ajuste de hiperparámetros utilizando validación cruzada para optimizar aún más el modelo de Random Forest. Este proceso resultó en una exactitud final del **93.75%**.

| Características PCA | Predicción | Probabilidad |
|--------------|------------|-------------|
| [-1286.519, -87.91, -19.43, -7.29, -4.98] | 1.0 | [0.0, 0.997, 0.0009, 0.0011] |
| [-521.319, -79.94, -9.92, -5.51, -1.90]  | 2.0 | [0.0, 0.003, 0.727, 0.269]   |

## Conclusión
- **Random Forest** resultó ser el modelo más efectivo para este problema, con la mayor exactitud y precisión entre las diferentes clases.
- **PCA** proporcionó una mejora en la interpretabilidad del modelo, aunque con una ligera disminución en la exactitud.
- El pipeline fue optimizado a través del **ajuste de hiperparámetros** y la validación cruzada, lo que llevó a un modelo fiable para la predicción de variedades de vino.
- Puede ver el resultado de los experimentos en el siguiente [Enlace](https://community.cloud.databricks.com/ml/experiments/1740077517975597?viewStateShareKey=3613330bab17ac06fdf3d97bbe14019422abb4e0c02ab12021213903dd2ee82d)

### Puede validar algo de los resultados en el siguiente apartado [Analisis de Resultados](https://github.com/aForero-GitHub/wine-classification-ml/blob/main/AnalisisResultados.md)

![PortadaAnalisis](https://github.com/aForero-GitHub/wine-classification-ml/blob/main/img/resultsAnalysis.png)
  
## Trabajo Futuro
1. Explorar métodos más avanzados de ensamble o redes neuronales.
2. Implementar técnicas más sofisticadas de ingeniería de características.
3. Investigar la interpretabilidad del modelo utilizando SHAP o LIME.

## Cómo Ejecutar Este Proyecto
1. Crear una cuenta gratuita en Databricks Community Edition: [Databricks CE](https://t.ly/IjRUp)
2. Clonar este repositorio.
3. Cargar el conjunto de datos y ejecutar el notebook en Databricks

 o localmente usando Spark y Python.
4. Monitorear los experimentos y el rendimiento del modelo utilizando **MLflow**.

## Contacto
Si tienes alguna pregunta o comentario, no dudes en contactarme vía [correo electrónico](foreromartinez.andres@gmail.com).
**LinkedIn**: [LinkedIn Profile](https://www.linkedin.com/in/andres-david-forero-martinez)

---

# --ENG--

# Wine Classification Using Machine Learning Models in Databricks

## Project Overview
This project is aimed at classifying wine varieties based on their chemical properties using different machine learning models. The dataset used for this classification problem comes from the UCI Machine Learning Repository's [Wine dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data). The data contains 13 chemical properties for three types of wine cultivars, and the goal is to predict the type of wine based on these features.

The project utilizes **Spark** and **Python** for data processing, model training, and evaluation, as these technologies are known for their speed, flexibility, and ability to handle large datasets. Databricks Community Edition is used to manage experiments, ensure traceability, and monitor the performance of models using **MLflow**.

## Dataset
- **Source**: [Wine Dataset from UCI](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data)
- **Description**: The dataset consists of 178 instances of wines grown in the same region in Italy but derived from three different cultivars. The dataset contains 13 chemical attributes (such as Alcohol, Malic acid, Ash, etc.) and a class label representing the type of wine (Class 1, Class 2, or Class 3).
- **Attributes**:
  1. Alcohol
  2. Malic acid
  3. Ash
  4. Alcalinity of ash  
  5. Magnesium
  6. Total phenols
  7. Flavanoids
  8. Nonflavanoid phenols
  9. Proanthocyanins
  10. Color intensity
  11. Hue
  12. OD280/OD315 of diluted wines
  13. Proline

## Objectives
1. **Data Exploration**: Perform an exploratory data analysis (EDA) to understand the dataset, identify relationships between variables, and detect outliers or missing values.
2. **Data Preprocessing**: Handle missing values, detect and treat outliers using the **IQR method**, and prepare the dataset for model training.
3. **Model Training**: Train multiple machine learning models, including:
   - Logistic Regression
   - Decision Tree
   - Random Forest
4. **Model Evaluation**: Evaluate the models based on accuracy, precision, recall, F1-score, and AUC-ROC (adapted for multiclass classification).
5. **Best Model Selection**: Identify the best-performing model and explain the criteria for selecting this model.
6. **Dimensionality Reduction**: Apply **Principal Component Analysis (PCA)** to reduce the dimensionality of the data and observe its impact on the model's performance.
7. **Hyperparameter Tuning**: Use cross-validation and hyperparameter tuning to optimize the Random Forest model.

## Tools and Libraries Used
- **Spark**: For distributed data processing and model training.
- **Python**: For scripting and custom logic.
- **Pandas**: For data manipulation.
- **ydata_profiling**: To generate an initial exploratory data report.
- **Matplotlib/Seaborn**: For visualizations.
- **MLflow**: To track experiments and model metrics.
- **Databricks Community Edition**: For experiment logging, training, and tracking.

## Data Preprocessing
- **Outlier Detection**: Outliers were identified using the **Interquartile Range (IQR) method**. Outliers were handled to improve model robustness.
  - **IQR Calculation**: For each feature, the IQR was calculated as Q3 - Q1, and data points falling below Q1 - 1.5*IQR or above Q3 + 1.5*IQR were considered outliers.
  - **Imputation**: Outliers were imputed using the median value of the respective feature to mitigate the impact of extreme values.

## Model Selection
Three models were trained to solve the classification problem:

1. **Logistic Regression**: A simple model that is often effective for multiclass classification problems.
2. **Decision Tree**: A model capable of capturing complex decision boundaries and handling non-linear relationships.
3. **Random Forest**: A robust ensemble model known for its ability to reduce overfitting and handle high-dimensional data.

**Why these models?**
- **Logistic Regression** is often a baseline for multiclass problems and provides easy interpretability.
- **Decision Tree** handles non-linearity and requires less data preparation.
- **Random Forest** aggregates multiple decision trees, providing high accuracy and robustness.

### Metrics Used for Evaluation
The following metrics were used to evaluate model performance:
- **Accuracy**: Overall correctness of the model.
- **Precision, Recall, F1-score**: For evaluating class-specific performance.
- **AUC-ROC**: Adapted for multiclass problems using one-vs-rest methodology.
- **Confusion Matrix**: Visualized to assess where the models performed well and where they struggled.

### Best Model Results: Random Forest
After training and comparing all three models, **Random Forest** showed the best performance with an accuracy of **96.88%**. Below are the results for two new samples:

| Features                                                            | Prediction | Probability           |
|---------------------------------------------------------------------|------------|-----------------------|
| [13.72, 1.43, 2.5, 16.7, 108, 3.4, 3.67, 0.19, 2.04, 6.8, 0.89, 2.87, 1285] | 1.0        | [0.0, 1.0, 0.0, 0.0]  |
| [12.37, 0.94, 1.36, 10.6, 88, 1.98, 0.57, 0.28, 0.42, 1.95, 1.05, 1.82, 520]  | 2.0        | [0.0, 0.0, 0.75, 0.25]|

## Dimensionality Reduction with PCA
**PCA** was applied to reduce dimensionality. However, this resulted in a slight drop in accuracy to **90.62%**. This is expected as some variance in the data is lost during dimensionality reduction.

## Hyperparameter Tuning and Cross-Validation
Finally, hyperparameter tuning using cross-validation was applied to further optimize the Random Forest model. This process resulted in a final accuracy of **93.75%**.

| PCA Features | Prediction | Probability |
|--------------|------------|-------------|
| [-1286.519, -87.91, -19.43, -7.29, -4.98] | 1.0 | [0.0, 0.997, 0.0009, 0.0011] |
| [-521.319, -79.94, -9.92, -5.51, -1.90]  | 2.0 | [0.0, 0.003, 0.727, 0.269]   |

## Conclusion
- **Random Forest** proved to be the most effective model for this problem, with the highest accuracy and precision across the different classes.
- **PCA** provided some improvement in model performance for interpretability but resulted in a slight drop in accuracy.
- The pipeline was optimized through **hyperparameter tuning** and cross-validation, leading to a reliable model for predicting wine varieties.
- Review the result experiments in the next [Link](https://community.cloud.databricks.com/ml/experiments/1740077517975597?viewStateShareKey=3613330bab17ac06fdf3d97bbe14019422abb4e0c02ab12021213903dd2ee82d)

### You can validate results in this link [results analysis](https://github.com/aForero-GitHub/wine-classification-ml/blob/main/AnalisisResultados.md)
  
## Future Work
1. Explore more advanced ensemble methods or neural networks.
2. Implement more sophisticated feature engineering techniques.
3. Investigate model interpretability using SHAP or LIME.

## How to Run This Project
1. Create a free Databricks Community Edition account: [Databricks CE](https://t.ly/IjRUp)
2. Clone this repository.
3. Load the dataset and execute the notebook in Databricks or locally using Spark and Python.
4. Track experiments and model performance using **MLflow**.

## Contact
For any questions or feedback, feel free to reach out via [email](foreromartinez.andres@gmail.com).
**LinkedIn**: [LinkedIn Profile](https://www.linkedin.com/in/andres-david-forero-martinez)

---
