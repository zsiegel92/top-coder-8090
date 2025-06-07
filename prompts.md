Tasks:
- The random forest model in `model.py` may not be very good - it's hard to tell. If you can make the model better AND have reason to believe your changes will not overfit, please do that. Do not implement cross-validation for now, we will do that later.
- The tree plot looks pretty bad. Too many nodes to actually interpret it. I'm not sure whether the model  Can you run `python model.py skip-training` a bunch of times with changes to `inspect_model`, each time looking at `decision_tree.png`, until the plot is more useful? If not possible because the tree is too big to easily visualize, then just tell me. Do NOT make the model simpler just so that we can visualize it more easily.
- Add two more plots:
  - basic y-versus-y-hat line plot, you can use seaborn
  - Three Umap plots (you may have to use uv to add some packages) where the input space is mapped onto 2d and the color indicates:
    - the true y
    - the yhat values
    - the difference y-yhat