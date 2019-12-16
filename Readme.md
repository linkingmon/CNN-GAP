# CNN GAP
### About
* This work is related to Physionet 2017 contest. (https://physionet.org/content/challenge-2017/1.0.0/)
* This work aim at classifying ECG signals into 4-types, including "Atrial Fibrillation", "Normal", "Other Diseases" and "Noisy".
* We reproduced works of the paper "Towards Understanding ECG Rhythm Classification Using Convolutional Neural Networks and Attention Mappings" by GoodFellow et. al.
### Our Works
* Train on N/A/O 3 type and got F1 score 0.84. (same as the paper)
* Observe CAM on ECG of each type and find what the model have learned.
* Train N/A/O/W 4 type with a fine-tuning procedure on the model pretrained by 3 type, and got F1 score 0.81.
* Fine-tuning the previous model with F1 loss and got F1 score 0.82, which is higher than all of the work.
* Analyze convergence property of F1 loss and WCE.
* Analyze convergence property of Softmax and Log-Softmax activation.

### Files discription
* Net/ : All models network structure
* utils/ : Preprocess function and loss function
* report/ : All .ppt of this work
* dataset/ : Dataset 
* jupyter notebook/Plot confusion matrix.ipynb : Plot confusion matrix of a model
* jupyter notebook/plot.py : Plot accuracy, loss or F1 score of the training process
* jupyter notebook/analyze f1 loss.ipynb : Analysis of convergence speed of F1, recall and precision.
* jupyter notebook/Train* : Training models
* model_description.txt : Description of all models