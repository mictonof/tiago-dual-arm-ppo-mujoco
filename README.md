# Aplicación de PPO en los brazos robóticos del robot TIAGo

<p align="center">
  <img src="docs/media/tiago_demo.gif" alt="Demostración del robot TIAGo" width="700"/>
</p>

Este proyecto explora la aplicación de **Proximal Policy Optimization (PPO)** sobre los brazos robóticos del **TIAGo** en simulación con **MuJoCo**, comenzando por tareas de **reaching** y avanzando hacia escenarios de **pick and place colaborativo**. El repositorio reúne el entorno, scripts de entrenamiento y evaluación, utilidades de visualización y carpetas de resultados generadas durante los experimentos.

https://github.com/user-attachments/assets/83ec4f58-bef6-43ac-b2f8-f14443ae21a5


## Tabla de contenido

- [1. Descripción general](#1-descripción-general)
- [2. Objetivos del proyecto](#2-objetivos-del-proyecto)
- [3. Tecnologías utilizadas](#3-tecnologías-utilizadas)
- [4. Estructura del proyecto](#4-estructura-del-proyecto)
- [5. Explicación de carpetas y archivos](#5-explicación-de-carpetas-y-archivos)
- [6. Flujo general de trabajo](#6-flujo-general-de-trabajo)
- [7. Descripción técnica del entorno](#7-descripción-técnica-del-entorno)
- [8. Entrenamiento con PPO](#8-entrenamiento-con-ppo)
- [9. Visualización y evaluación](#9-visualización-y-evaluación)
- [10. Resultados generados por el proyecto](#10-resultados-generados-por-el-proyecto)
- [11. Cómo ejecutar el proyecto](#11-cómo-ejecutar-el-proyecto)
- [12. Trabajo futuro](#12-trabajo-futuro)


---

## 1. Descripción general

El propósito de este repositorio es investigar cómo aplicar **aprendizaje por refuerzo profundo** al control de los brazos del robot **TIAGo**, usando un entorno de simulación física. La lógica principal del proyecto se centra en tres ideas:

1. **Reaching**: mover el efector final hacia un punto objetivo 3D.
2. **Selección automática del brazo**: cuando hay dos brazos disponibles, usar el que esté geométricamente más cerca del objetivo.
3. **Manipulación avanzada**: extender la tarea hacia esquemas de **pick and place** y, más adelante, a manipulación colaborativa entre ambos brazos.

---

## 2. Objetivos del proyecto

### Objetivo general
Implementar y entrenar políticas basadas en PPO para que el robot TIAGo pueda resolver tareas de control continuo con sus brazos en simulación.

### Objetivos específicos
- Definir un entorno de reaching dual-arm en MuJoCo.
- Ubicar correctamente el efector final del gripper.
- Generar objetivos aleatorios dentro de una región de trabajo.
- Diseñar espacios de estado, acción y recompensa.
- Entrenar políticas con PPO y evaluar su comportamiento.
- Sentar la base para tareas más complejas como grasping y pick and place colaborativo.

---

## 3. Tecnologías utilizadas

- **Python**
- **MuJoCo**
- **Gymnasium**
- **Stable-Baselines3**
- **NumPy**
- **TensorBoard**
- **robot_descriptions / tiago++_mj_description**

---

## 4. Estructura del proyecto

> La siguiente estructura resume la organización esperada del repositorio según el desarrollo del proyecto. Si en tu versión final algún nombre cambia, solo ajusta la sección correspondiente del README.

```bash
tiago_mujoco/
├── .venv/
├── tiago_dual_arm_reach_env.py
├── train_dual_auto_arm_ppo.py
├── eval_dual_auto_arm_ppo.py
├── view_dual_floor_mouse_plane.py
├── smoke_test_dual_auto_arm.py
├── diagnose_tiago_arm_env.py
├── diagnostics/
│   ├── right_arm_actuators.csv
│   ├── goal_distribution.csv
│   ├── actuator_sensitivity.csv
│   └── approx_workspace.csv
├── checkpoints_dual/
├── best_dual_model/
├── eval_logs_dual/
├── models/
├── logs/
├── docs/
│   └── media/
│       └── tiago_demo.gif
├── README.md
└── requirements.txt
```

---

## 5. Explicación de carpetas y archivos

### `.venv/`
Entorno virtual de Python del proyecto.

**Qué contiene**
- Paquetes instalados.
- Ejecutables del entorno virtual.
- Dependencias específicas del proyecto.

**Por qué es importante**
Permite aislar la instalación de librerías como MuJoCo, Stable-Baselines3 y Gymnasium del resto del sistema.

---

### `tiago_dual_arm_reach_env.py`
Archivo principal del entorno de aprendizaje por refuerzo.

**Responsabilidad**
- Define la clase del entorno del robot.
- Carga el modelo del TIAGo en MuJoCo.
- Genera el goal aleatorio.
- Selecciona el brazo activo.
- Define observaciones, acciones, recompensa y condición de éxito.

**Es el archivo más importante del proyecto**, porque ahí se formaliza el problema de control como un entorno RL.

---

### `train_dual_auto_arm_ppo.py`
Script de entrenamiento con PPO.

**Responsabilidad**
- Crea los entornos de entrenamiento y evaluación.
- Configura los hiperparámetros de PPO.
- Lanza el proceso de aprendizaje.
- Guarda checkpoints y el mejor modelo.

**Qué hace en la práctica**
Convierte la formulación del entorno en una política entrenable.

---

### `eval_dual_auto_arm_ppo.py`
Script de evaluación del modelo entrenado.

**Responsabilidad**
- Carga el modelo guardado.
- Reinicia episodios del entorno.
- Ejecuta la política en modo determinista.
- Imprime métricas como distancia al goal y éxito o fracaso.

**Uso típico**
Se usa después del entrenamiento para comprobar si el robot realmente aprendió un comportamiento útil.

---

### `view_dual_floor_mouse_plane.py`
Script de visualización interactiva.

**Responsabilidad**
- Muestra el robot en MuJoCo.
- Permite visualizar el prisma de muestreo, el plano central y los puntos de referencia.
- Facilita inspeccionar la posición del goal y de los efectores finales.

**Valor dentro del proyecto**
Es clave para depurar visualmente si el goal está bien ubicado, si el efector final tiene sentido y si el espacio de trabajo está correctamente definido.

---

### `smoke_test_dual_auto_arm.py`
Prueba rápida de integridad del entorno.

**Responsabilidad**
- Carga el entorno.
- Ejecuta un reset.
- Verifica que los componentes básicos funcionen.
- Sirve para detectar errores tempranos antes de entrenar.

---

### `diagnose_tiago_arm_env.py`
Script de diagnóstico técnico del robot y del entorno.

**Responsabilidad**
- Inspecciona actuadores.
- Analiza sensibilidad de articulaciones.
- Aproxima el workspace del gripper.
- Estudia cómo se distribuyen los goals.

**Por qué es relevante**
Este archivo fue especialmente útil para entender:
- qué hace cada actuador,
- hasta dónde llega el gripper,
- si el goal está dentro del espacio alcanzable,
- y dónde estaban los errores geométricos del entorno.

---

## Carpetas de resultados y análisis

### `diagnostics/`
Carpeta donde se guardan resultados de diagnóstico en formato CSV.

#### `right_arm_actuators.csv`
Contiene información sobre los actuadores del brazo derecho, como rangos de control y nombres.

#### `goal_distribution.csv`
Registra la distribución de goals generados tras múltiples resets del entorno.

#### `actuator_sensitivity.csv`
Muestra cómo responde el efector final cuando se perturba cada actuador.

#### `approx_workspace.csv`
Aproxima el workspace del gripper a partir de múltiples muestras.

**Valor**
Esta carpeta documenta la parte experimental de depuración geométrica del proyecto.

---

### `checkpoints_dual/`
Carpeta de checkpoints intermedios del entrenamiento.

**Qué contiene**
- Modelos guardados periódicamente durante el entrenamiento.

**Por qué sirve**
Permite:
- reanudar entrenamiento,
- comparar iteraciones,
- rescatar modelos aunque el entrenamiento final falle.

---

### `best_dual_model/`
Carpeta que guarda el mejor modelo según evaluación.

**Qué contiene**
- `best_model.zip` y archivos relacionados.

**Por qué es importante**
El último modelo entrenado no siempre es el mejor. Esta carpeta conserva el modelo con mejor desempeño validado.

---

### `eval_logs_dual/`
Registros de evaluación periódica durante el entrenamiento.

**Qué contiene**
- resultados de episodios de evaluación,
- métricas por iteración,
- información usada para decidir cuál fue el mejor modelo.

---

### `models/`
Carpeta opcional para modelos exportados manualmente o versiones finales.

**Uso sugerido**
Guardar variantes como:
- modelos finales,
- modelos experimentales,
- versiones entrenadas en distintas configuraciones.

---

### `logs/`
Carpeta opcional para logs generales y TensorBoard.

**Qué suele incluir**
- archivos de logging del entrenamiento,
- curvas de recompensa,
- estadísticas de PPO.

---

### `docs/`
Carpeta de documentación del proyecto.

**Uso sugerido**
- diagramas,
- capturas del robot,
- gráficas de entrenamiento,
- documentación adicional para presentaciones o informes.

---

### `docs/media/`
Subcarpeta para recursos visuales.

**Ejemplo**
- GIF del robot para el encabezado del README.
- imágenes del workspace.
- screenshots del visor de MuJoCo.

---

### `requirements.txt`
Archivo recomendado para listar dependencias del proyecto.

**Ejemplo de contenido**
```txt
numpy
gymnasium
mujoco
stable-baselines3
tensorboard
robot_descriptions
```

---

### `README.md`
Archivo principal de documentación.

**Propósito**
- explicar qué hace el proyecto,
- cómo está organizado,
- cómo ejecutarlo,
- y qué significa cada carpeta.

---

## 6. Flujo general de trabajo

El flujo técnico del proyecto puede resumirse así:

1. **Construcción del entorno**
   - carga del robot,
   - definición del efector final,
   - definición del goal,
   - selección del brazo.

2. **Diagnóstico geométrico**
   - análisis de actuadores,
   - verificación del workspace,
   - inspección del espacio de muestreo.

3. **Entrenamiento**
   - uso de PPO,
   - ajuste de hiperparámetros,
   - almacenamiento de checkpoints.

4. **Evaluación**
   - prueba del mejor modelo,
   - inspección visual del comportamiento,
   - revisión de distancias al goal.

5. **Refinamiento**
   - modificación de reward,
   - modificación del prisma,
   - ajuste del efector final,
   - nuevas corridas de entrenamiento.

---

## 7. Descripción técnica del entorno

### Problema resuelto
El entorno busca resolver una tarea de **reaching dual-arm**.

### Goal
Se define un objetivo aleatorio dentro de un prisma 3D.

### Efector final
El efector final no se dejó en una referencia arbitraria. Se definió como el **centro de acción del gripper**, usando los puntos funcionales de los dedos.

### Selección del brazo
Para cada goal, se calcula qué efector está más cerca. Ese brazo queda activo.

### Espacio de estados
La observación incluye:
- posiciones articulares,
- velocidades articulares,
- posición del efector final,
- posición del goal,
- error goal-efector,
- indicador del brazo activo.

### Espacio de acciones
La acción se definió como un vector continuo de 7 dimensiones, correspondiente a las 7 articulaciones del brazo activo.

### Reward
La recompensa suele combinar:
- penalización por distancia,
- recompensa por progreso,
- penalización por acción excesiva,
- bonus de éxito.

---

## 8. Entrenamiento con PPO

PPO fue utilizado porque:
- es adecuado para espacios de acción continuos,
- tiene estabilidad razonable,
- y permite trabajar bien con control articular.

### Componentes relevantes
- **Actor**: propone acciones.
- **Critic**: estima el valor del estado.
- **Clipping**: evita actualizaciones bruscas.

### Hiperparámetros típicos
- `learning_rate`
- `n_steps`
- `batch_size`
- `gamma`
- `gae_lambda`
- `clip_range`
- `target_kl`

---

## 9. Visualización y evaluación

### Visualización
El visor permite:
- observar el goal,
- mostrar el prisma de muestreo,
- mostrar los efectores finales,
- inspeccionar la coherencia geométrica del problema.

### Evaluación
Durante la evaluación se revisa:
- distancia final al goal,
- brazo activo elegido,
- consistencia del movimiento,
- tasa de éxito.

---

## 10. Resultados generados por el proyecto

El proyecto genera distintos tipos de resultados:

### Resultados numéricos
- rewards,
- success rate,
- pérdida del modelo,
- distancias finales.

### Resultados visuales
- capturas del robot,
- GIFs,
- inspección del espacio de trabajo.

### Resultados de diagnóstico
- sensibilidad de actuadores,
- distribución del goal,
- aproximación del workspace.

---

## 11. Cómo ejecutar el proyecto

### 1. Activar entorno virtual
```bash
cd ~/projects/tiago_mujoco
source .venv/bin/activate
```

### 2. Ejecutar prueba rápida
```bash
python smoke_test_dual_auto_arm.py
```

### 3. Ejecutar diagnóstico
```bash
python diagnose_tiago_arm_env.py
```

### 4. Entrenar
```bash
python train_dual_auto_arm_ppo.py
```

### 5. Visualizar
```bash
export MUJOCO_GL=glfw
python view_dual_floor_mouse_plane.py
```

### 6. Evaluar
```bash
export MUJOCO_GL=glfw
python eval_dual_auto_arm_ppo.py
```

---

## 12. Trabajo futuro

Líneas de continuación sugeridas:
- mejorar el reward shaping,
- ampliar el entorno hacia tareas de pregrasp y grasp,
- incorporar pick and place colaborativo,
- usar curriculum learning,
- evaluar otras políticas además de PPO,
- integrar cinemática inversa o control híbrido en fases avanzadas.

---

## Nota final

Este repositorio no solo documenta un entrenamiento con PPO, sino también el proceso de **ingeniería del entorno**, que fue fundamental para lograr comportamientos coherentes en el robot. La definición del efector final, el tamaño del espacio de muestreo y la formulación geométrica del problema fueron tan importantes como el propio algoritmo de aprendizaje.

