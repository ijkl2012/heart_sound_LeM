A small two-class classification code package for congenital heart disease (CHD), including the model, testing and evaluation scripts, common metric computations, and IG-based interpretability visualization.

model_training.py: Defines and builds the model (including custom front-end and modules) for training/inference use.
evaluate_test.py: Loads the test set and multiple pre-trained fold models, performs inference, and outputs results.
metrics.py: Utility functions for common classification metrics and confusion matrix, enabling unified evaluation of predictions.
vis_ig.py: Integrated Gradients (IG) interpretability analysis, with segmented coloring to visualize regions the model focuses on.
train_val_splits.csv and test.csv: Fixed data split lists (IDs for training/validation/test samples).



