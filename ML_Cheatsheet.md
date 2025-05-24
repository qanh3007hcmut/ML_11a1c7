# MACHINE LEARNING COMPREHENSIVE CHEATSHEET

## Table of Contents
1. [Bayesian Networks](#bayesian-networks)
2. [Hidden Markov Models (HMM)](#hidden-markov-models)
3. [Support Vector Machines (SVM)](#support-vector-machines)
4. [Principal Component Analysis (PCA)](#principal-component-analysis)
5. [Random Forests (RF)](#random-forests)
6. [Gradient Boosting](#gradient-boosting)
7. [Logistic Regression](#logistic-regression)
8. [Conditional Random Fields (CRF)](#conditional-random-fields)

---

## Bayesian Networks

### Key Concepts
- Directed acyclic graph (DAG) representing conditional dependencies
- Each node is conditionally independent of non-descendants given its parents
- Joint probability factorizes as: $P(X_1,X_2,...,X_n) = \prod_{i=1}^{n} P(X_i|Parents(X_i))$

### Steps for Probability Calculation
1. Identify conditional independence from graph
2. Apply chain rule with independence assumptions
3. Substitute known values from conditional probability tables

### Example (from mock exam)
Network structure:
```
Previous_Purchases → Promotion_Clicked → Purchase_Highlighted_Product
                             ↓                       ↑
Time_on_Site_min    →   Viewed_Highlighted_Product_Details
```

**Joint probability calculation:**
$P(A,B,C,D,E) = P(A) \times P(B) \times P(C|A) \times P(D|B,C) \times P(E|C,D)$

where:
- A = Previous_Purchases > 5
- B = Time_on_Site_min > 30
- C = Promotion_Clicked = Yes
- D = Viewed_Highlighted_Product_Details = Yes
- E = Purchase_Highlighted_Product = Yes

**Numerical Example:**
$P(A=1,B=1,C=1,D=1,E=1) = 0.3 \times 0.6 \times 0.7 \times 0.8 \times 0.9 = 0.09072$

### Tips
- Draw the graph first to visualize dependencies
- Break complex queries into simpler conditional probabilities
- Use d-separation to identify conditional independence

---

## Hidden Markov Models

### Key Components
- **Hidden states**: Unobservable states the system can be in (e.g., Low/Medium/High engagement)
- **Observable emissions**: Actual observations produced by states (e.g., user actions)
- **Transition probabilities**: $P(S_t|S_{t-1})$ - Probability of transitioning between states
- **Emission probabilities**: $P(O_t|S_t)$ - Probability of observing output given the state
- **Initial state probabilities**: $P(S_1)$ - Probability of starting in each state

### Key Algorithms
1. **Forward Algorithm**: Computes probability of observation sequence
2. **Viterbi Algorithm**: Finds most likely state sequence given observations
3. **Baum-Welch Algorithm**: Learns model parameters from observations (EM algorithm)

### Forward Algorithm
**Purpose**: Calculate $P(O_1,...,O_T)$ - probability of entire observation sequence

**Notation**:
- $P(S_1 = i)$ - Initial probability of state i
- $P(O_t|S_t = i)$ - Probability of observation at time t given state i
- $P(S_t = j|S_{t-1} = i)$ - Transition probability from state i to j

**Steps**:
- **Initialization**: 
  $P(O_1, S_1 = i) = P(S_1 = i) \times P(O_1|S_1 = i)$

- **Recursion**: 
  $P(O_1,...,O_t, S_t = j) = P(O_t|S_t = j) \times \sum_{i=1}^{N} [P(O_1,...,O_{t-1}, S_{t-1} = i) \times P(S_t = j|S_{t-1} = i)]$

- **Termination**: 
  $P(O_1,...,O_T) = \sum_{i=1}^{N} P(O_1,...,O_T, S_T = i)$

### Viterbi Algorithm
**Purpose**: Find most likely state sequence $S_1,...,S_T$ given observations $O_1,...,O_T$

**Steps**:
- **Initialization**:
  $V_1(i) = P(S_1 = i) \times P(O_1|S_1 = i)$

- **Recursion**:
  $V_t(j) = P(O_t|S_t = j) \times \max_i[V_{t-1}(i) \times P(S_t = j|S_{t-1} = i)]$

- **Path backtracking**: 
  Keep track of best previous state for each current state and time step

### Example (from mock exam)
**States**: Low (L), Medium (M), High (H) Engagement  
**Emissions**: V (view product), A (add to cart), P (click promotion)

**Given**:
- Initial state: Medium Engagement with P(S₁ = M) = 1.0
- Transition probabilities:
  - P(S_t = L|S_{t-1} = M) = 0.2
  - P(S_t = M|S_{t-1} = M) = 0.5
  - P(S_t = H|S_{t-1} = M) = 0.3
- Emission probabilities:
  - P(V|S_t = M) = 0.4
  - P(A|S_t = L) = 0.1
  - P(A|S_t = M) = 0.4
  - P(A|S_t = H) = 0.5
- Observation: O = {V, A}

**Forward Algorithm Calculation**:

**Step 1:** Compute forward probabilities for t=1  
$P(O_1 = V, S_1 = M) = P(S_1 = M) \times P(O_1 = V|S_1 = M)$  
$P(O_1 = V, S_1 = M) = 1.0 \times 0.4 = 0.4$

**Step 2:** Compute forward probabilities for t=2  
For state L:  
$P(O_1 = V, O_2 = A, S_2 = L) = P(O_2 = A|S_2 = L) \times \sum_i [P(O_1 = V, S_1 = i) \times P(S_2 = L|S_1 = i)]$  
$P(O_1 = V, O_2 = A, S_2 = L) = P(A|L) \times [P(O_1 = V, S_1 = M) \times P(S_2 = L|S_1 = M)]$  
$P(O_1 = V, O_2 = A, S_2 = L) = 0.1 \times [0.4 \times 0.2] = 0.1 \times 0.08 = 0.008$

For state M:  
$P(O_1 = V, O_2 = A, S_2 = M) = P(O_2 = A|S_2 = M) \times [P(O_1 = V, S_1 = M) \times P(S_2 = M|S_1 = M)]$  
$P(O_1 = V, O_2 = A, S_2 = M) = 0.4 \times [0.4 \times 0.5] = 0.4 \times 0.2 = 0.08$

For state H:  
$P(O_1 = V, O_2 = A, S_2 = H) = P(O_2 = A|S_2 = H) \times [P(O_1 = V, S_1 = M) \times P(S_2 = H|S_1 = M)]$  
$P(O_1 = V, O_2 = A, S_2 = H) = 0.5 \times [0.4 \times 0.3] = 0.5 \times 0.12 = 0.06$

**Total probability**:  
$P(O_1 = V, O_2 = A) = P(O_1 = V, O_2 = A, S_2 = L) + P(O_1 = V, O_2 = A, S_2 = M) + P(O_1 = V, O_2 = A, S_2 = H)$  
$P(O_1 = V, O_2 = A) = 0.008 + 0.08 + 0.06 = 0.148$

**Viterbi Algorithm Calculation**:

**Step 1:** Compute Viterbi values for t=1  
$V_1(M) = P(S_1 = M) \times P(O_1 = V|S_1 = M) = 1.0 \times 0.4 = 0.4$

**Step 2:** Compute Viterbi values for t=2  
$V_2(L) = P(O_2 = A|S_2 = L) \times \max_i[V_1(i) \times P(S_2 = L|S_1 = i)]$  
$V_2(L) = 0.1 \times [V_1(M) \times P(S_2 = L|S_1 = M)] = 0.1 \times [0.4 \times 0.2] = 0.008$

$V_2(M) = P(O_2 = A|S_2 = M) \times \max_i[V_1(i) \times P(S_2 = M|S_1 = i)]$  
$V_2(M) = 0.4 \times [V_1(M) \times P(S_2 = M|S_1 = M)] = 0.4 \times [0.4 \times 0.5] = 0.08$

$V_2(H) = P(O_2 = A|S_2 = H) \times \max_i[V_1(i) \times P(S_2 = H|S_1 = i)]$  
$V_2(H) = 0.5 \times [V_1(M) \times P(S_2 = H|S_1 = M)] = 0.5 \times [0.4 \times 0.3] = 0.06$

Since $V_2(M) = 0.08$ is the highest value, the most likely path is:  
**Viterbi path**: Medium → Medium

### Tips
- The forward algorithm sums over all possible paths (total probability)
- The Viterbi algorithm finds the single most likely path
- Forward values are often denoted as α and Viterbi values as δ in literature
- For numerical stability, calculations are often done in log space

---

## Support Vector Machines

### Key Concepts
- Finds optimal hyperplane maximizing margin between classes
- Support vectors are points closest to the decision boundary
- Can handle non-linear boundaries using kernel functions

### Linear SVM
- Decision boundary: $w^T x + b = 0$
- Classification: $f(x) = sign(w^T x + b)$
- Optimization objective: Minimize $\frac{1}{2}||w||^2$ subject to $y_i(w^T x_i + b) \geq 1$

### Kernel Functions
1. **Linear**: $K(x_i,x_j) = x_i^T x_j$
2. **Polynomial**: $K(x_i,x_j) = (x_i^T x_j + c)^d$
3. **RBF/Gaussian**: $K(x_i,x_j) = \exp(-\gamma ||x_i-x_j||^2)$

### Example: Polynomial Kernel
Given vectors $s_1=[5,10]$ and $s_2=[30,60]$, degree $d=3$:

**Step 1:** Compute dot product 
$s_1 \cdot s_2 = 5 \times 30 + 10 \times 60 = 150 + 600 = 750$

**Step 2:** Apply polynomial formula with $d=3$
$K(s_1,s_2) = (s_1 \cdot s_2 + 1)^3 = 751^3 = 423,564,751$

### Example: RBF Kernel
Given vectors $s_1=[5,10]$ and $s_2=[30,60]$, $\gamma=0.001$:

**Step 1:** Calculate squared Euclidean distance
$||s_1-s_2||^2 = (30-5)^2 + (60-10)^2 = 25^2 + 50^2 = 625 + 2500 = 3125$

**Step 2:** Apply RBF formula
$K(s_1,s_2) = \exp(-0.001 \times 3125) = \exp(-3.125) \approx 0.0439$

### Example: Linear Classifier
Given hyperplane $0.5x_1 - 0.08x_2 + 5 = 0$, classify point P5(6,100):
$0.5(6) - 0.08(100) + 5 = 3 - 8 + 5 = 0$

Since result is 0, P5 lies exactly on the hyperplane.

### Tips
- Linear kernel for linearly separable data
- Polynomial/RBF for non-linear boundaries
- RBF is most flexible but can overfit
- Support vectors determine the boundary

---

## Principal Component Analysis

### Key Concepts
- Dimensionality reduction technique
- Maps data to a lower-dimensional space preserving variance
- Principal components are eigenvectors of covariance matrix

### Algorithm Steps
1. Center the data (subtract mean)
2. Compute covariance matrix: $C = \frac{1}{n}X^TX$
3. Find eigenvalues and eigenvectors of covariance matrix: $V_i^{-1} C V_i = \lambda_i$
   - $\lambda_i$ are eigenvalues, $V_i$ are eigenvectors
   - $$\det(A - \lambda I) = 0$$ 
   - $$\det(A) = |A| = ad - bc$$
   - $$\det(A) = a_{11}(a_{22}a_{33} - a_{23}a_{32}) - a_{12}(a_{21}a_{33} - a_{23}a_{31}) + a_{13}(a_{21}a_{32} - a_{22}a_{31})$$
   - $$(A - \lambda_i I)\vec{v}_i = \vec{0}$$
4. Select top k eigenvectors (with largest eigenvalues)
5. Project data onto selected eigenvectors

### Formulas
- **Variance explained by component i**: $\frac{\lambda_i}{\sum_{j=1}^n \lambda_j}$
- **Total variance explained by first k components**: $\frac{\sum_{i=1}^k \lambda_i}{\sum_{j=1}^n \lambda_j}$
- **Projection**: $Z = X V$ (where V contains eigenvectors as columns)

### PCA via SVD
- Compute SVD: $X = U \Sigma V^T$
- V contains principal components (eigenvectors)
- Singular values $\sigma_i$ relate to eigenvalues: $\lambda_i = \frac{\sigma_i^2}{n-1}$

### Example (from mock exam)
Given covariance matrix and eigenvalues/vectors:
```
Eigenvalues:
λ1=135.9, v1=[0.27,0.89,0.37]
λ2=29.3, v2=[0.48,-0.45,0.75]
λ3=9.8, v3=[0.83,-0.10,-0.55]
```

**Percentage variance explained by PC1:**
$\frac{\lambda_1}{\lambda_1+\lambda_2+\lambda_3} = \frac{135.9}{135.9+29.3+9.8} = \frac{135.9}{175.0} = 77.7\%$

**Percentage variance explained by PC1+PC2:**
$\frac{\lambda_1+\lambda_2}{\lambda_1+\lambda_2+\lambda_3} = \frac{135.9+29.3}{175.0} = 94.4\%$

### Interpreting Principal Components
PC1 [0.27,0.89,0.37] means:
- Strongest weight on 2nd feature (0.89)
- Moderate weights on 1st and 3rd features
- Represents "general size" or main variation pattern

### Tips
- Standardize data if features are on different scales
- Look for "elbow" in scree plot to select number of components
- Interpret components based on feature loadings (weights)

---

## Random Forests

### Key Concepts
- Ensemble of decision trees using bagging
- Combines predictions from multiple trees to reduce variance
- Uses random feature selection at each split

### Algorithm
1. Create bootstrap sample of training data
2. Train decision tree on bootstrap sample
3. At each node, select random subset of features
4. Repeat to create multiple trees
5. Aggregate predictions by majority vote (classification) or averaging (regression)

### Variance Reduction Mechanisms
1. **Bootstrap Aggregation (Bagging)**:
   - Each tree trained on different random subset
   - Reduces variance by averaging predictions

2. **Random Feature Selection**:
   - Consider only subset of features at each split
   - Decorrelates trees, further reducing variance

### Example (from mock exam)
Given 200 trees in Random Forest model:
- 130 trees predict "Yes" (purchase)
- 70 trees predict "No" (no purchase)

**Majority Vote Decision**: "Yes" (since 130 > 70)
**Probability Estimate**: 130/200 = 0.65 (65% probability of purchase)

### Adjusting Threshold
- Default threshold: 0.5 (predict "Yes" if ≥ 0.5)
- Lower threshold (e.g., 0.3): Increases sensitivity, reduces false negatives
- Higher threshold (e.g., 0.7): Increases precision, reduces false positives

### Tips
- More trees generally better (but diminishing returns)
- mtry parameter (features per split) impacts correlation between trees
- OOB (out-of-bag) error provides internal validation
- Feature importance can be calculated from tree statistics

---

## Gradient Boosting

### Key Concepts
- Sequential ensemble method (trees built in sequence)
- Each tree corrects errors made by previous trees
- Optimizes loss function via gradient descent in function space

### Algorithm
1. Initialize model with constant prediction (average target value)
2. For m = 1 to M:
   a. Compute residuals (negative gradients)
   b. Train weak learner (e.g., shallow tree) to predict residuals
   c. Update model: $F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$

### Key Parameters
- **Learning rate (η)**: Controls contribution of each tree (shrinkage)
- **Number of trees**: Total ensemble size
- **Tree depth**: Controls complexity of each weak learner

### Example (from mock exam)
Initial prediction F₀(x) = 0.25 (average purchase probability)
Learning rate η = 0.1

For customer session A with actual purchase = 1:
- Initial residual: 1 - 0.25 = 0.75
- First tree prediction h₁(A) = 0.60
- Updated prediction: F₁(A) = 0.25 + 0.1 × 0.60 = 0.31
- New residual: 1 - 0.31 = 0.69

For session B with actual purchase = 0:
- Initial residual: 0 - 0.25 = -0.25
- First tree prediction h₁(B) = -0.20
- Updated prediction: F₁(B) = 0.25 + 0.1 × (-0.20) = 0.23
- New residual: 0 - 0.23 = -0.23

### Learning Rate Trade-offs
- **Small η (e.g., 0.01-0.1)**:
  - Slower convergence
  - Better generalization
  - Requires more trees

- **Large η (e.g., 0.5-1.0)**:
  - Faster convergence
  - Risk of overfitting
  - May converge to suboptimal solution

### Tips
- Start with small learning rate and many trees
- Early stopping based on validation set
- Consider regularization techniques (column/row subsampling)
- Tree depth of 3-5 often sufficient for weak learners

---

## Logistic Regression

### Key Concepts
- Linear model for binary classification
- Predicts probability using logistic (sigmoid) function
- Uses maximum likelihood estimation for training

### Formulas
- **Logistic function**: $\sigma(z) = \frac{1}{1+e^{-z}}$
- **Linear predictor**: $z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n$
- **Probability model**: $P(y=1|x) = \sigma(z) = \frac{1}{1+e^{-z}}$
- **Log-odds (logit)**: $\log\left(\frac{P(y=1|x)}{1-P(y=1|x)}\right) = \beta_0 + \beta_1 x_1 + ... + \beta_n x_n$

### Coefficient Interpretation
- **Continuous variables**: For each unit increase in x₁, log-odds increases by β₁
- **Binary variables**: Log-odds differs by β₁ between groups
- **Exponentiated coefficients**: $e^{\beta_1}$ is the odds ratio per unit increase in x₁

### Example (from mock exam)
Coefficients:
- Intercept: -3.0
- pages_viewed: 0.05
- time_on_site_min: 0.02
- previous_purchases: 0.1
- cart_value: 0.01
- device_type=Mobile: -0.5
- promotion_clicked=Yes: 1.5

For customer with values:
- pages_viewed = 20
- time_on_site_min = 30
- previous_purchases = 3
- cart_value = $70
- device_type = Mobile
- promotion_clicked = Yes

**Step 1: Calculate log-odds**
$z = -3.0 + 0.05(20) + 0.02(30) + 0.1(3) + 0.01(70) - 0.5 + 1.5$
$z = -3.0 + 1.0 + 0.6 + 0.3 + 0.7 - 0.5 + 1.5 = 0.6$

**Step 2: Convert to probability**
$P(purchase=1) = \frac{1}{1+e^{-0.6}} = \frac{1}{1+0.5488} = 0.6457$ (64.57%)

### Tips
- Standardize continuous predictors
- Check for multicollinearity
- Interpret coefficients as affecting log-odds
- Categorical predictors need reference category
- Calculate predicted probabilities for interpretation

---

## Conditional Random Fields

### Key Concepts
- Discriminative sequence model (vs generative like HMM)
- Models conditional probability P(Y|X) directly
- Can incorporate arbitrary, overlapping features

### Formulas
- **Model form**: $P(Y|X) = \frac{1}{Z(X)} \exp\left(\sum_{t=1}^{T} \sum_{k=1}^{K} w_k f_k(y_{t-1}, y_t, X, t)\right)$
- **Partition function**: $Z(X) = \sum_{Y} \exp\left(\sum_{t=1}^{T} \sum_{k=1}^{K} w_k f_k(y_{t-1}, y_t, X, t)\right)$
- **Feature functions**: $f_k(y_{t-1}, y_t, X, t)$ can depend on:
  - Current state $y_t$
  - Previous state $y_{t-1}$
  - Entire observation sequence X
  - Position t

### Types of Features
1. **State features**: $f(y_t, X, t)$ - depend on current state and observations
2. **Transition features**: $f(y_{t-1}, y_t)$ - depend on adjacent states

### Example (from mock exam)
**Feature functions**:
- f₁(y(t),x(t),t) = 1 if y(t)=CP AND x_t = is_highlighted_product, else 0. (w₁=1.2)
- f₂(y(t),x(t),t) = 1 if y(t)=CP AND x_t = time_on_page > 60, else 0. (w₂=0.8)
- f₃(y(t-1),y(t)) = 1 if y(t-1)=NCP AND y(t)=CP, else 0. (w₃=-0.5)
- f₄(y(t-1),y(t)) = 1 if y(t-1)=CP AND y(t)=CP, else 0. (w₄=0.9)

For sequence (y₁=NCP, y₂=CP) with:
- Page 1: Not highlighted, time = 45s
- Page 2: Highlighted, time = 70s

**Active features at t=1**: None (no features are active)

**Active features at t=2**:
- f₁(CP,highlighted,2) = 1 → w₁ × 1 = 1.2
- f₂(CP,time>60,2) = 1 → w₂ × 1 = 0.8
- f₃(NCP,CP) = 1 → w₃ × 1 = -0.5

**Unnormalized score**: 0 + (1.2 + 0.8 - 0.5) = 1.5

### Vs HMMs
- CRFs: Discriminative (P(Y|X)), can use arbitrary features
- HMMs: Generative (P(X,Y)), limited by independence assumptions

### Tips
- CRFs excel when rich, overlapping features are available
- Good for NLP, sequence labeling tasks
- Training more complex than HMM
- Can incorporate domain knowledge via feature engineering

---

## Model Evaluation

### Classification Metrics
- **Accuracy**: $(TP + TN) / (TP + TN + FP + FN)$
- **Precision**: $TP / (TP + FP)$ - How many predicted positives are correct
- **Recall/Sensitivity**: $TP / (TP + FN)$ - How many actual positives are caught
- **F1-Score**: $2 \times \frac{Precision \times Recall}{Precision + Recall}$
- **AUC-ROC**: Area under Receiver Operating Characteristic curve

### When to Use Each
- **Accuracy**: Balanced classes, equal error costs
- **Precision**: Cost of false positives high (e.g., spam filtering)
- **Recall**: Cost of false negatives high (e.g., disease detection)
- **F1-Score**: Need balance between precision and recall
- **AUC-ROC**: Model comparison, ranking performance

### Example Scenario (E-commerce)
- **Goal**: Predict who will buy highlighted product
- **High Precision**: Focus marketing budget efficiently
- **High Recall**: Don't miss potential customers
- **AUC**: Overall model discrimination power

### Tips
- Use appropriate metric for business objective
- Consider class imbalance
- Threshold tuning affects precision-recall trade-off
- Cross-validation for reliable estimates