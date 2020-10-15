# TODO
- use [Group KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
- add logic to handle starting training from checkpoints
- add weight to loss with [BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
- determine the [best threshold value](https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/) for determining smoke/no-smoke
- configure [TTA](https://towardsdatascience.com/test-time-augmentation-tta-and-how-to-perform-it-with-keras-4ac19b67fb4d)