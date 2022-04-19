# IngredientExtractor

Used FastText vectorization and 1D Keras CNN model to classify text word-wise and 2-grams.

- **01_analyzedata.ipynb** : Notebook to analyze data
- ***02_classifier_model.ipynb** : Train and Test Keras Classifier
- **03_inference_callable.ipynb** : File to run inference from saved model. (production ready code)

# Results

Accuracy: 0.91
Macro Avg Accuracy:  0.80

Classification report can be seen in notebook 02.

Production ready code is on notebook 03 which calls method from IngredientExtractor.py file.
