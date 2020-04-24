Linear Regresssion from scratch

## Dataset
The simple cengage systolic blood pressure dataset:  
https://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/frame.html

## Purpose
Gain familiarity with the algorithm by developing it from scratch.
Hence, best ML practices such as train/test/cross-validation splits 
are NOT prioritized.

#### Run verified with python 3.7 and numpy 1.18.1
python Runner.py

## Visualize
![Blood Pressure for Age And Weight](plots/visualize.png)

### Min-Max Normalized
(data - min) / (max - min)  
Puts feature data and labels within [0, 1] range  

![BP for Age and Weight Normailzed](plots/min_max_normalized.png)

## Training Learning Curve

![Training Learning Curve](plots/training_learning_curve.png)

## Trained Model

![Trained Model](plots/trained_model.png)

## Normal Equation Trained Model

![Normal Equation Trained Model](plots/norm_equation_model.png)

## Actual vs Projected

![Actual vs Projected](plots/actual_vs_projected.png)

## Normal Equation Actual vs Projected

![Normal Equation Actual vs Projected](plots/norm_equation_actual_vs_projected.png)
