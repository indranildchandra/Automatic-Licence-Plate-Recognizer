# Automatic-Licence-Plate-Recognizer
Automatic-Licence-Plate-Recognizer

## Files to run:
- Step 1: Run pre_processing_pipeline_research.ipynb
- Step 2: Run creating_dataset_pipeline.ipynb
- Step 3: Run training_pipeline_research.ipynb
- Step 4: Run model_training_pipeline.ipynb
- Step 5: Run model_prediction_pipeline.ipynb

## High Level Approach:
1. Create a pre-processing pipeline to segment out individual characters from the License Plate. Problems - non-uniform lighting (resolved using dynamic thresholding), skewed images (resolved using Hough transform), image blurring (have not solved yet). Note that since the provided data already had license plates cropped from the original scene there was no need to build a License Plate Detection pipeline (TF object detector or EAST text detector). Refer to pre_processing_pipeline_research.ipynb for reference.
2. Consider only the subset of the dataset where exactly one target bounding box is derived for each character in the annotation provided for each datapoint. (Success Rate = 24%, this denotes the ability of the pre-processing pipeline to identify exactly one target bounding box per annotation character and not whether the target bounding box is accurate or not). Refer to creating_dataset_pipeline.ipyb for reference.
3. Since the size of the dataset was too small to be considered for building a CNN, I could have either augmented more data or use a readily available characters' dataset derived from License plates and re-used it. I proceeded with the second approach. All of the data samples provided were used only for testing purposes whereas the alternate additional data was used only for training purposes to check the generalization measure of the model. Refer to creating_dataset_pipeline.ipyb for reference.
4. Hyperopt and Hyperas were used to find out the optimal hyperparameters for the proposed CNN architecture5. Refer to training_pipeline_research.ipyb for reference.
5. CNN model was finally trained with the derived optimal hyperparameters with 10-fold Cross-Validation strategy with shuffling of data. Refer to model_training_pipeline.ipyb for reference.
6. Model performance was checked on the test data segmented out in Step #3. Refer to model_prediction_pipeline.ipynb for reference.
