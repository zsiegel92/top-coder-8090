Tasks:

- The random forest model in `model.py` may not be very good - it's hard to tell. If you can make the model better AND have reason to believe your changes will not overfit, please do that. Do not implement cross-validation for now, we will do that later.
- The tree plot looks pretty bad. Too many nodes to actually interpret it. I'm not sure whether the model Can you run `python model.py skip-training` a bunch of times with changes to `inspect_model`, each time looking at `decision_tree.png`, until the plot is more useful? If not possible because the tree is too big to easily visualize, then just tell me. Do NOT make the model simpler just so that we can visualize it more easily.
- Add two more plots:
  - basic y-versus-y-hat line plot, you can use seaborn
  - Three Umap plots (you may have to use uv to add some packages) where the input space is mapped onto 2d and the color indicates:
    - the true y
    - the yhat values
    - the difference y-yhat

---

As you go, use `pyright <filename>` to ensure you are generating statically correct code. You can run any of these files with `python <filename>` (should use the virtual environment that is already activated in this terminal) and install packages with `uv` if needed.

---

Now we are going to implement 5-fold cross-validation with our model architecture to choose hyperparameters without overfitting.

- Identify the most important hyperparameters (e.g. n_estimators and anything else that might make a random forest model perform differently). Make those arguments to a function, maybe the `train` function or maybe a new function - organize things in the cleanest, best way you possibly can.
- Your job will be to train and run the model with 5-fold cross-validation across a range of hyperparameters and whichever architecture performs.
- Refactor so that you can cleanly run an experiment where you compare each hyperparameter set (with 5-fold cross-validation, each with the same fractional train-validation-test split) and generate a spreadsheet that makes it clear which is the best one!

---

So which one is the best? What did we learn? Did they perform vastly differently or not really?

Okay you say "the improvement is modest" but obviously over this training set training on the whole thing should be better, the goal is to make it perform better on unseen data! Am I missing something?

```md
The top-performing hyperparameters achieved R² = 0.9192 ±
0.0141:
{
'n_estimators': 500,
'max_depth': None,
'min_samples_split': 5,
'min_samples_leaf': 4,
'max_features': None,
'bootstrap': True
}

Key Insights

1. More Trees = Better Performance

- 500 estimators performed best
- 200+ estimators consistently outperformed 50-100 trees
- Diminishing returns suggest 500 is near optimal

2. Deeper Trees Work Better

- max_depth=None (unlimited) was optimal
- Limited depth (10, 20, 30) consistently underperformed
- The data benefits from complex decision boundaries

3. Regularization Helps

- min_samples_leaf=4 was optimal (vs default 1)
- min_samples_split=5 outperformed default 2
- These prevent overfitting by requiring more samples per
  decision

4. Feature Selection Matters

- max_features=None (use all features) was best
- With only 3 features total, restricting to 'sqrt' (1-2
  features) hurt performance
- Small feature space benefits from using all available
  information

5. Bootstrap is Essential

- All bootstrap=False experiments failed (OOB score
  incompatibility)
- Bootstrap sampling provides ensemble diversity

Performance Improvement

- Baseline model: R² = 0.9165, MSE = 18,408
- Optimized model: R² = 0.9192, MSE = 17,766
- Improvement: +0.27% R², -3.5% MSE

The improvement is modest because the original model was
already very good. The hyperparameter optimization
provided a statistically significant but small
performance gain while ensuring we're using the best
possible configuration.
```

---

Rather than `dict` and `update`, make things more type-safe - you can sacrifice a little convenience for this. Make some sort of `RandomForestHyperParameters` Pydantic model (in `models_and_constants.py`) and pass the parameters to `RandomForestRegressor` each explicitly

---

Now we are going to make `model.py` a little bit more agnostic to model architecture - make some of the hyperparameters determine which _type_ of model we are using and compare the random forest to one or two other reasonable model choices, e.g. a dense MLP. Different model architectures should each have their own pydantic model representing their hyperparameters and the input to `train` should be something like `RandomForestHyperParameters | MLPHyperParameters`. If it simplifies things for both of those to inherit from a shared class to make it easier to use some shared fields to organize runs, that's cool! Define models in the `models_and_constants.py` file.

Refactor in that way and then run an experiment where you compare each model (with cross-validation) and generate a spreadsheet that makes it clear which is the best one! Save all models to a subdirectory called `model_weights/`, give `load_model` an easy parameter to choose which one (probably it should take one of those HyperParameters objects meentioned above) and then change `predict` to load the "best" model and use it!

---

Finally, remove all comments you added to the code. Remove docstrings as well. Clean up the CLI interface, make sure the model dumping/loading has clean, obvious entrypoints. If it helps to make a separate file called `experiments.py`, please do that! I want to ship this to production. While you are editing the `model.py` file, do not break the behavior of `predict(...)` - that is actually being important and used by another process runing in a loop. But you can change anything else about the file.

---

Okay can you find the 20 examples in our training data for which this model performs the worst? These are "edge cases". Store them in a CSV called `edge_cases.csv`.
