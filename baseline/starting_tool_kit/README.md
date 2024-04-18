## Welcome to IUGC 2024!

##### In this repository, we will provide you some code to help you get quick start in our competition.

Below, we will introduce the function of files and codes in this repository.

In ***starting_tool_kit***, we provide training-related codes for you.

-   ***dataset_sample***: It is a folder containing both pos and neg samples for training.
-   ***utils***: It contains a lot of codes to help you training your model.
    -   **augmentation.py**: It contains two Class to achieve Data Augmentation in *Classification* and *Segmentation* tasks.
    -   **criterion.py**: You can design your own combination of diverse loss functions, it will be used in training.
    -   **dataset_classification.py**: It provides a Class to Read data for *Classification* task.
    -   **dataset_segmentation.py**: It provides a Class to Read **labeled** data for *Segmentation* task.
    -   **evaluator.py**: It contains three Class to evaluate your model. Each class receives tow numpy arrays (prediction and label), you can use process function to make full assessment of your model.
    -   **loss.py**: It contains a lot of common loss functions.
    -   **metric.py:**   It contains a *Metric Class* to help you get validation result on this metric during training. It also play a role in finding the best model during training.
-   ***training_seg.py:***  It is a template code for training segmentation model. But it should be noticed that our challenge have three tasks. It is possible to train model for classification and segmentation  at the same time. Thus, maybe you can modify it.