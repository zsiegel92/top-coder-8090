Tasks:
- The random forest model in `model.py` may not be very good - it's hard to tell. If you can make the model better AND have reason to believe your changes will not overfit, please do that. Do not implement cross-validation for now, we will do that later.
- The tree plot looks pretty bad. Too many nodes to actually interpret it. I'm not sure whether the model  Can you run `python model.py skip-training` a bunch of times with changes to `inspect_model`, each time looking at `decision_tree.png`, until the plot is more useful? If not possible because the tree is too big to easily visualize, then just tell me. Do NOT make the model simpler just so that we can visualize it more easily.
- Add two more plots:
  - basic y-versus-y-hat line plot, you can use seaborn
  - Three Umap plots (you may have to use uv to add some packages) where the input space is mapped onto 2d and the color indicates:
    - the true y
    - the yhat values
    - the difference y-yhat
  

---


Now we are going to implement 5-fold cross-validation with our model architecture to choose hyperparameters without overfitting.

- Identify the most important hyperparameters (e.g. n_estimators and anything else that might make a random forest model perform differently). Make those arguments to a function, maybe the `train` function or maybe a new function - organize things in the cleanest, best way you possibly can.
- Your job will be to train and run the model with 5-fold cross-validation across a range of hyperparameters and whichever architecture performs.
- Refactor so that you can cleanly run an experiment where you compare each hyperparameter set (with 5-fold cross-validation, each with the same fractional train-validation-test split) and generate a spreadsheet that makes it clear which is the best one!





---

Now we are going to make `model.py` a little bit more agnostic to model architecture - make some of the hyperparameters determine which *type* of model we are using and compare the random forest to one or two other reasonable model choices, e.g. a dense MLP. Different model architectures should each have their own pydantic model representing their hyperparameters and the input to `train` should be something like `RandomForestHyperParameters | MLPHyperParameters`. If it simplifies things for both of those to inherit from a shared class to make it easier to use some shared fields to organize runs, that's cool! Define models in the `models_and_constants.py` file.

Refactor in that way and then run an experiment where you compare each model (with cross-validation) and generate a spreadsheet that makes it clear which is the best one! Save all models to a subdirectory called `model_weights/`, give `load_model` an easy parameter to choose which one (probably it should take one of those HyperParameters objects meentioned above) and then change `predict` to load the "best" model and use it!