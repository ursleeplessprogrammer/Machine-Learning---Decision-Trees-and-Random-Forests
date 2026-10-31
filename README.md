# 🌳 Decision Trees and Random Forests

> **From one branching set of decisions to a whole forest of them.**  
> Learn how tree-based models make predictions, why forests combine many trees, and how to build both with Python.


## 🗺️ Trail Map

- [What Is a Decision Tree?](#-what-is-a-decision-tree)
- [How a Tree Makes Decisions](#-how-a-tree-makes-decisions)
- [Classification and Regression Trees](#-classification-and-regression-trees)
- [How Trees Choose Splits](#-how-trees-choose-splits)
- [Overfitting and Pruning](#️-overfitting-and-pruning)
- [What Is a Random Forest?](#-what-is-a-random-forest)
- [How a Random Forest Works](#-how-a-random-forest-works)
- [Decision Trees vs. Random Forests](#-decision-trees-vs-random-forests)
- [Training and Evaluation](#-training-and-evaluation)
- [Python Examples](#-python-examples)
- [Feature Importance](#-feature-importance)
- [Strengths and Limitations](#️-strengths-and-limitations)
- [When to Use Them](#-when-to-use-them)
- [Field Notes](#-field-notes)
- [Further Reading](#-further-reading)

## 🔦 What Is a Decision Tree?

A **decision tree** is a machine-learning model that makes predictions by asking a series of questions about the input data.

For example, a model that predicts whether a customer will renew a subscription might ask:

```text
Has the customer used the product this month?
├── Yes → Is the subscription close to renewal?
│   ├── Yes → Predict likely to renew
│   └── No  → Predict likely to renew
└── No → Predict unlikely to renew
```

Each question is a **split**. Each split sends an example down a branch until it reaches a **leaf**, where the tree produces a prediction.

Decision trees can be used for:

- **Classification:** Predict a category, such as `approved` or `declined`.
- **Regression:** Predict a number, such as a price or delivery time.

## 🌿 How a Tree Makes Decisions

A tree consists of:

- **Root node:** The first decision point.
- **Internal nodes:** Questions or feature tests.
- **Branches:** Outcomes of those tests.
- **Leaf nodes:** The final prediction.

A tree is built by repeatedly selecting a feature and a split point that separates the training examples in a useful way. The process continues until a stopping condition is reached, such as a maximum depth or a minimum number of records in a leaf.

A tree is easy to picture, but a large tree can still be difficult to understand. Limiting its depth can make its decisions more readable.

## 🏷️ Classification and Regression Trees

### Classification trees

A classification tree predicts a class. At each leaf, it typically predicts the class most common among training examples that reached that leaf. It can also provide class scores or probabilities based on the examples in the leaf.

Example:

```text
Input: [age = 25, visits = 12]
Prediction: likely_to_renew
```

### Regression trees

A regression tree predicts a numerical value. A leaf commonly predicts an average of the training targets that reached it.

Example:

```text
Input: [size = 1200 sq ft, bedrooms = 2]
Prediction: estimated price = $240,000
```

In scikit-learn, `DecisionTreeClassifier` and `DecisionTreeRegressor` are the corresponding estimators. ([scikit-learn decision trees](https://scikit-learn.org/stable/modules/tree.html))

## ⚖️ How Trees Choose Splits

A tree evaluates possible splits and chooses one that improves the separation of the data according to a criterion.

For classification, common criteria include:

- **Gini impurity:** Measures how mixed the classes are in a node.
- **Entropy / information gain:** Measures uncertainty and how much a split reduces it.

A pure node contains examples from one class. A split is useful when it produces child nodes that are less mixed than the parent node.

For regression, tree splits are commonly chosen to reduce variation in the target values within the resulting leaves.

The exact split objective depends on the estimator and its settings. Scikit-learn documents supported criteria for its tree classifiers and regressors. ([DecisionTreeClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html))

## ⚠️ Overfitting and Pruning

A tree can keep splitting until it memorizes details of its training data. This is **overfitting**: the model performs well on training data but less well on new examples.

Common controls include:

- `max_depth`: Limits how many levels the tree can grow.
- `min_samples_split`: Sets the minimum examples required to split a node.
- `min_samples_leaf`: Sets the minimum examples allowed in a leaf.
- `max_leaf_nodes`: Limits the number of leaves.
- `ccp_alpha`: Controls cost-complexity pruning.

A shallow tree may be easier to explain but too simple to capture important patterns. A deep tree may fit training data closely but generalize poorly. Choose these settings using validation data or cross-validation—not by repeatedly tuning against the final test set.

## 🌲 What Is a Random Forest?

A **random forest** is an ensemble of decision trees. Instead of relying on a single tree, it trains many trees with variation in the data and features they consider.

For classification, the forest combines the trees’ predictions, typically by voting. For regression, it averages their predictions.

The variation among trees helps reduce the instability of an individual decision tree. Scikit-learn describes a random forest as fitting trees on subsamples and averaging their predictions to improve predictive accuracy and control overfitting. ([RandomForestClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html))

## 🔄 How a Random Forest Works

A simplified random-forest training process is:

1. Create a varied sample of the training data for a tree, commonly using bootstrap sampling.
2. Grow a decision tree on that sample.
3. At each split, consider a random subset of features.
4. Repeat to build many trees.
5. Combine the trees’ predictions.

```text
Training data
   ├── Tree 1 ──┐
   ├── Tree 2 ──┼── Combine predictions → Forest prediction
   ├── Tree 3 ──┤
   └── ...      ┘
```

Each tree may make a different prediction. Combining them often produces a more stable result than using one unconstrained tree.

## ⚖️ Decision Trees vs. Random Forests

| Feature | Decision tree | Random forest |
|---|---|---|
| Model structure | One tree | Many trees |
| Ease of visualization | Often straightforward when shallow | Usually harder to inspect as a whole |
| Stability | Can change substantially with small data changes | Often more stable because predictions are combined |
| Predictive performance | Can be a useful baseline | Often stronger on tabular tasks, but must be tested |
| Training and prediction | Usually faster and smaller | More computation and storage may be needed |
| Interpretability | Individual paths can be explained | Combined decisions are less directly interpretable |

A random forest is not guaranteed to outperform a single tree on every dataset. Compare them on the same evaluation split.

## 🧪 Training and Evaluation

A typical supervised-learning workflow is:

1. Separate the features (`X`) from the target (`y`).
2. Split data into training and test sets.
3. Fit models using training data.
4. Tune settings using validation data or cross-validation.
5. Evaluate once on held-out test data.
6. Inspect errors and decide whether the models are suitable.

For classification, useful metrics include:

- **Precision:** Of predicted positive examples, how many were actually positive?
- **Recall:** Of actual positive examples, how many did the model find?
- **F1-score:** A combined precision/recall measure.
- **Confusion matrix:** Shows counts of correct and incorrect classifications.
- **ROC-AUC or PR-AUC:** Evaluates ranking behavior across thresholds; PR-AUC can be useful when the positive class is rare.

For regression, common metrics include MAE, RMSE, and R².

Avoid data leakage: fit preprocessing steps using training data only, and use splits that respect time or groups when the data requires it.

## 🧑‍💻 Python Examples

Install the dependencies:

```bash
pip install scikit-learn
```

### Decision tree classifier

This example uses scikit-learn’s built-in breast cancer dataset to demonstrate classification. It is for learning only and is not a medical diagnostic tool.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

tree = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=5,
    random_state=42,
)

tree.fit(X_train, y_train)
predictions = tree.predict(X_test)

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=dataset.target_names,
    )
)
```

### Random forest classifier

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

forest = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

forest.fit(X_train, y_train)
predictions = forest.predict(X_test)

print(
    classification_report(
        y_test,
        predictions,
        target_names=dataset.target_names,
    )
)
```

The dataset is provided only to demonstrate the code. Its results should not be treated as evidence of performance in another setting.

### Visualize a small tree

```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(18, 10))
plot_tree(
    tree,
    feature_names=dataset.feature_names,
    class_names=dataset.target_names,
    filled=True,
    rounded=True,
    max_depth=3,
)
plt.tight_layout()
plt.show()
```

A full tree can become too large to read. Limiting the displayed depth helps show its structure without changing the trained model.

## 🔍 Feature Importance

Tree models can report **impurity-based feature importance**, which summarizes how much features reduced impurity across the fitted trees.

Treat this as a model diagnostic, not proof that a feature causes the outcome. Impurity-based importance can be misleading—for example, it can favor high-cardinality features. Scikit-learn documents permutation importance as an alternative: shuffle one feature and measure how much model performance drops on a chosen dataset. ([scikit-learn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html))

Use held-out data for feature-importance analysis when possible, and interpret correlated features carefully.

## ⚠️ Strengths and Limitations

### Strengths

- Can model nonlinear relationships and feature interactions.
- Requires less feature scaling than many linear or distance-based methods.
- A shallow decision tree can be inspected as a set of rules.
- Random forests often provide a strong baseline for structured, tabular data.
- Can support both classification and regression.

### Limitations

- A single tree can overfit and be unstable.
- Deep trees can be difficult to interpret.
- Random forests are less transparent than one small tree.
- Forests can require more memory and computation.
- Impurity-based feature importance can be biased or misunderstood.
- Neither model automatically understands that correlation implies causation.
- Performance can degrade when new data differs substantially from training data.

## 🧭 When to Use Them

### Consider a decision tree when:

- You want an interpretable baseline.
- The task can be explained with a small set of rules.
- You want to explore nonlinear patterns and feature interactions.
- You can control tree size and evaluate generalization.

### Consider a random forest when:

- You want to compare an ensemble against a single tree.
- The data is structured and tabular.
- A more stable model is useful.
- You can accept less direct interpretability and more computation.

Always compare models with a suitable baseline and evaluation design. If interpretability or latency is critical, those requirements should influence the choice.

## 📝 Field Notes

- A decision tree predicts by following feature-based splits to a leaf.
- Trees can handle classification and regression.
- Deep trees can overfit; limits and pruning help control complexity.
- A random forest combines many varied trees.
- Forests are often more stable than individual trees, but are harder to inspect directly.
- Feature importance is useful for investigation, not a causal explanation.
- Choose evaluation metrics that reflect the actual task and the costs of mistakes.

## 🔗 Further Reading

- [scikit-learn: Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [scikit-learn: DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [scikit-learn: RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [scikit-learn: Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
- [scikit-learn: Permutation Feature Importance](https://scikit-learn.org/stable/modules/permutation_importance.html)

---

*Decision trees make predictions through a sequence of splits. Random forests build on that idea by combining many trees—trading some interpretability for a model that is often more stable.*
