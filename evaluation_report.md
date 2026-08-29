# AuraVision - Model Evaluation Report

## Evaluation Summary

AuraVision was evaluated using a held-out test set of 112 images across seven image-quality classes.

|Model|Accuracy|
|-|-:|
|Random Forest|96.43%|
|CNN|77.68%|

## Random Forest

The classical Random Forest classifier achieved an accuracy of 96.43% on the test set.

The weighted precision was 96.94%, weighted recall was 96.43%, and weighted F1-score was 96.24%.

Strong performance was observed for blurry, noisy, overexposed, and underexposed images. The model also detected corrupted and defective samples with high recall.

## CNN

The convolutional neural network achieved an accuracy of 78.57% on the same evaluation set.

The CNN provides complementary spatial analysis and supports Grad-CAM visualization for model interpretability.

## Evaluated Classes

The evaluation covered the following seven classes:

* Acceptable
* Blurry
* Corrupted
* Defective
* Noisy
* Overexposed
* Underexposed

## Conclusion

The evaluation demonstrates that AuraVision's hybrid classical and deep-learning approach can identify multiple forms of image degradation. The Random Forest branch provides strong classification performance, while the CNN branch provides complementary visual analysis and Grad-CAM explainability.

