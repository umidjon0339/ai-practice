# Scikit-learn Course — Real-Life Examples

A progressive course. Do the lessons **in order** — each one builds on the previous.

Run every script with the project venv:

```bash
../.venv/bin/python lesson_folder/script.py
# or activate first:
source ../.venv/bin/activate
```

## The Learning Path

| # | Lesson | Real-life problem | What you learn |
|---|--------|-------------------|----------------|
| 1 | `01_regression_house_prices` | Predict house prices | Features/target, train/test split, LinearRegression, MAE & R² |
| 2 | `02_classification_cancer` | Diagnose tumors (real medical data) | LogisticRegression, accuracy vs precision vs recall, confusion matrix |
| 3 | `03_preprocessing_churn` | Predict customer churn from messy data | Missing values, categorical encoding, scaling, Pipeline & ColumnTransformer |
| 4 | `04_random_forest_loans` | Approve or reject loan applications | Decision trees, RandomForest, overfitting, feature importance |
| 5 | `05_model_selection` | Pick the best churn model honestly | Cross-validation, GridSearchCV, why one test split lies to you |
| 6 | `06_clustering_customers` | Segment mall customers for marketing | KMeans, elbow method, silhouette score, PCA visualization |
| 7 | `07_spam_filter_project` | Full end-to-end SMS spam filter | Text features (TF-IDF), Naive Bayes, saving/loading a whole pipeline |

## The 5 ideas that make you "get" scikit-learn

1. **Everything is an estimator.** Every model follows the same API:
   `model.fit(X, y)` → learn, `model.predict(X)` → answer, `model.score(X, y)` → grade.
   Learn it once, use every algorithm.

2. **X is always 2D, y is 1D.** X = table of samples × features. y = one answer per sample.
   90% of beginner errors are shape errors.

3. **Never evaluate on data the model trained on.** That's like grading students
   on the exact questions they memorized. Hence `train_test_split` and cross-validation.

4. **Preprocessing is part of the model.** Scalers and encoders must be fit on
   training data only, then applied to test data. `Pipeline` makes this automatic
   and leak-proof.

5. **Metrics depend on the problem.** Accuracy is meaningless for rare events
   (99% of emails are not spam → "always say not-spam" is 99% accurate and useless).

## Which model for which real-life problem? (cheat sheet)

| You want to… | Task type | Start with |
|---|---|---|
| Predict a number (price, temperature, sales) | Regression | `LinearRegression`, then `RandomForestRegressor` |
| Predict a category (spam/ham, sick/healthy, churn/stay) | Classification | `LogisticRegression`, then `RandomForestClassifier` |
| Group similar things with no labels (customer types) | Clustering | `KMeans` |
| Compress many features for visualization | Dim. reduction | `PCA` |
| Work with text | Feature extraction | `TfidfVectorizer` + `MultinomialNB` |
