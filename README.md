# cpsc552project
Final project for CPSC 552 - adversarial noise in neural networks. 

### Summary

This report investigates the susceptibility of neural networks to adversarial attacks, specifically focusing on food classification tasks. Adversarial attacks involve subtle modifications to input data, leading to significant misclassification errors even in high-performing models. This vulnerability poses a significant risk in critical domains such as healthcare, where misclassification can have serious consequences.

Using the Food-101 dataset, which contains 101,000 images across 101 food categories, we aim to establish a robust neural network architecture for food classification. Our approach involves:
1. Developing a baseline model for food classification.
2. Demonstrating the baseline model's vulnerability to adversarial attacks.
3. Retraining the model with adversarial examples to enhance robustness.

Through this investigation, we find that the EfficientNet architecture, despite its high accuracy and mobile-friendly design, is susceptible to adversarial attacks. By retraining the model on adversarial examples, we achieve a more robust food classification model. Our results indicate that this approach can improve the resilience of neural networks against adversarial attacks in practical applications.

### Instructions 

Run the `Adversarial_Training.ipynb` notebook.