---
title: logitstic_knn
---


```python
from google.colab import drive
drive.mount('/content/gdrive/')
```

    Mounted at /content/gdrive/
    


```python
from sklearn import datasets
import numpy as np
import pandas as pd

```


```python
iris = datasets.load_iris()
```


```python
X = iris.data[:,[2,3]]
y = iris.target
print(np.unique(y))
```

    [0 1 2]
    


```python
from sklearn.model_selection import train_test_split
X_train , X_test , y_train, y_test = train_test_split(
    X,y,test_size=0.3, random_state=1 , stratify=y)

```


```python
print(np.bincount(y))
```

    [50 50 50]
    


```python
print(np.bincount(y_train))
```

    [35 35 35]
    


```python
print(np.bincount(y_test))
```

    [15 15 15]
    


```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
```


```python
from sklearn.linear_model import Perceptron

## random_state  == seed

ppn = Perceptron(max_iter=40,eta0=0.1, tol=1e-3, random_state=1)
ppn.fit(X_train_std, y_train)
```




    Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=0.1,
               fit_intercept=True, max_iter=40, n_iter_no_change=5, n_jobs=None,
               penalty=None, random_state=1, shuffle=True, tol=0.001,
               validation_fraction=0.1, verbose=0, warm_start=False)




```python
y_pred = ppn.predict(X_test_std)
print(sum(y_test != y_pred))
```

    1
    


```python
from sklearn.metrics import accuracy_score
#accuracy
print(accuracy_score(y_test,y_pred))
```

    0.9777777777777777
    


```python
#ppn.score

print(ppn.score(X_test_std,y_test))
```

    0.9777777777777777
    


```python
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(X,y,classifier,test_idx=None,resolution=0.02):
  markers = ('s','x','o','^','v')
  colors = ('red','blue','lightgreen','gray','cyan')
  cmap = ListedColormap(colors[:len(np.unique(y))])

  x1_min, x1_max = X[:, 0].min() -1 , X[:,0].max() +1
  x2_min, x2_max = X[:,1].min()-1 , X[:,1].max() + 1
  xx1 , xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))

  Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
  Z = Z.reshape(xx1.shape)
  plt.contourf(xx1,xx2,Z,alpha=0.3,cmap=cmap)
  plt.xlim(xx1.min(),xx1.max())
  plt.ylim(xx2.min(),xx2.max())
  for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl,0],y=X[y==cl,1],alpha=0.8,c=colors[idx],marker=markers[idx],label=cl,edgecolor='black')
    if test_idx:
      X_test, y_test = X[test_idx, : ], y[test_idx]
      plt.scatter(X_test[:,0],X_test[:,1],facecolors='none',edgecolor='black',alpha=1.0,linewidths=1,marker='o',s=100,label='test set')


```


```python
X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined  = np.hstack((y_train,y_test))
plot_decision_regions(X=X_combined_std,y=y_combined,classifier=ppn,test_idx=range(105,150))
plt.xlabel('petal length [std]')
plt.ylabel('petal width [std]')
plt.tight_layout()
plt.show
```




    <function matplotlib.pyplot.show>




    
![png](/img/output_14_1.png)
    



```python
ppn.coef_
```




    array([[-0.10655204, -0.11836728],
           [ 0.31790327, -0.3670884 ],
           [ 0.36018414,  0.30003858]])




```python
ppn.coef_.dot(X_test_std[1])+ppn.intercept_
```




    array([ 0.14787666,  0.01315806, -1.13331509])




```python
y_test[1]
```




    0




```python
ppn.intercept_
```




    array([-1.00000000e-01, -2.77555756e-17, -4.00000000e-01])




```python
help(Perceptron)
```

    Help on class Perceptron in module sklearn.linear_model._perceptron:
    
    class Perceptron(sklearn.linear_model._stochastic_gradient.BaseSGDClassifier)
     |  Perceptron
     |  
     |  Read more in the :ref:`User Guide <perceptron>`.
     |  
     |  Parameters
     |  ----------
     |  
     |  penalty : {'l2','l1','elasticnet'}, default=None
     |      The penalty (aka regularization term) to be used.
     |  
     |  alpha : float, default=0.0001
     |      Constant that multiplies the regularization term if regularization is
     |      used.
     |  
     |  fit_intercept : bool, default=True
     |      Whether the intercept should be estimated or not. If False, the
     |      data is assumed to be already centered.
     |  
     |  max_iter : int, default=1000
     |      The maximum number of passes over the training data (aka epochs).
     |      It only impacts the behavior in the ``fit`` method, and not the
     |      :meth:`partial_fit` method.
     |  
     |      .. versionadded:: 0.19
     |  
     |  tol : float, default=1e-3
     |      The stopping criterion. If it is not None, the iterations will stop
     |      when (loss > previous_loss - tol).
     |  
     |      .. versionadded:: 0.19
     |  
     |  shuffle : bool, default=True
     |      Whether or not the training data should be shuffled after each epoch.
     |  
     |  verbose : int, default=0
     |      The verbosity level
     |  
     |  eta0 : double, default=1
     |      Constant by which the updates are multiplied.
     |  
     |  n_jobs : int, default=None
     |      The number of CPUs to use to do the OVA (One Versus All, for
     |      multi-class problems) computation.
     |      ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
     |      ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
     |      for more details.
     |  
     |  random_state : int, RandomState instance, default=None
     |      The seed of the pseudo random number generator to use when shuffling
     |      the data.  If int, random_state is the seed used by the random number
     |      generator; If RandomState instance, random_state is the random number
     |      generator; If None, the random number generator is the RandomState
     |      instance used by `np.random`.
     |  
     |  early_stopping : bool, default=False
     |      Whether to use early stopping to terminate training when validation.
     |      score is not improving. If set to True, it will automatically set aside
     |      a stratified fraction of training data as validation and terminate
     |      training when validation score is not improving by at least tol for
     |      n_iter_no_change consecutive epochs.
     |  
     |      .. versionadded:: 0.20
     |  
     |  validation_fraction : float, default=0.1
     |      The proportion of training data to set aside as validation set for
     |      early stopping. Must be between 0 and 1.
     |      Only used if early_stopping is True.
     |  
     |      .. versionadded:: 0.20
     |  
     |  n_iter_no_change : int, default=5
     |      Number of iterations with no improvement to wait before early stopping.
     |  
     |      .. versionadded:: 0.20
     |  
     |  class_weight : dict, {class_label: weight} or "balanced", default=None
     |      Preset for the class_weight fit parameter.
     |  
     |      Weights associated with classes. If not given, all classes
     |      are supposed to have weight one.
     |  
     |      The "balanced" mode uses the values of y to automatically adjust
     |      weights inversely proportional to class frequencies in the input data
     |      as ``n_samples / (n_classes * np.bincount(y))``
     |  
     |  warm_start : bool, default=False
     |      When set to True, reuse the solution of the previous call to fit as
     |      initialization, otherwise, just erase the previous solution. See
     |      :term:`the Glossary <warm_start>`.
     |  
     |  Attributes
     |  ----------
     |  coef_ : ndarray of shape = [1, n_features] if n_classes == 2 else         [n_classes, n_features]
     |      Weights assigned to the features.
     |  
     |  intercept_ : ndarray of shape = [1] if n_classes == 2 else [n_classes]
     |      Constants in decision function.
     |  
     |  n_iter_ : int
     |      The actual number of iterations to reach the stopping criterion.
     |      For multiclass fits, it is the maximum over every binary fit.
     |  
     |  classes_ : ndarray of shape (n_classes,)
     |      The unique classes labels.
     |  
     |  t_ : int
     |      Number of weight updates performed during training.
     |      Same as ``(n_iter_ * n_samples)``.
     |  
     |  Notes
     |  -----
     |  
     |  ``Perceptron`` is a classification algorithm which shares the same
     |  underlying implementation with ``SGDClassifier``. In fact,
     |  ``Perceptron()`` is equivalent to `SGDClassifier(loss="perceptron",
     |  eta0=1, learning_rate="constant", penalty=None)`.
     |  
     |  Examples
     |  --------
     |  >>> from sklearn.datasets import load_digits
     |  >>> from sklearn.linear_model import Perceptron
     |  >>> X, y = load_digits(return_X_y=True)
     |  >>> clf = Perceptron(tol=1e-3, random_state=0)
     |  >>> clf.fit(X, y)
     |  Perceptron()
     |  >>> clf.score(X, y)
     |  0.939...
     |  
     |  See also
     |  --------
     |  
     |  SGDClassifier
     |  
     |  References
     |  ----------
     |  
     |  https://en.wikipedia.org/wiki/Perceptron and references therein.
     |  
     |  Method resolution order:
     |      Perceptron
     |      sklearn.linear_model._stochastic_gradient.BaseSGDClassifier
     |      sklearn.linear_model._base.LinearClassifierMixin
     |      sklearn.base.ClassifierMixin
     |      sklearn.linear_model._stochastic_gradient.BaseSGD
     |      sklearn.linear_model._base.SparseCoefMixin
     |      sklearn.base.BaseEstimator
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, penalty=None, alpha=0.0001, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, eta0=1.0, n_jobs=None, random_state=0, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  __abstractmethods__ = frozenset()
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from sklearn.linear_model._stochastic_gradient.BaseSGDClassifier:
     |  
     |  fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None)
     |      Fit linear model with Stochastic Gradient Descent.
     |      
     |      Parameters
     |      ----------
     |      X : {array-like, sparse matrix}, shape (n_samples, n_features)
     |          Training data.
     |      
     |      y : ndarray of shape (n_samples,)
     |          Target values.
     |      
     |      coef_init : ndarray of shape (n_classes, n_features), default=None
     |          The initial coefficients to warm-start the optimization.
     |      
     |      intercept_init : ndarray of shape (n_classes,), default=None
     |          The initial intercept to warm-start the optimization.
     |      
     |      sample_weight : array-like, shape (n_samples,), default=None
     |          Weights applied to individual samples.
     |          If not provided, uniform weights are assumed. These weights will
     |          be multiplied with class_weight (passed through the
     |          constructor) if class_weight is specified.
     |      
     |      Returns
     |      -------
     |      self :
     |          Returns an instance of self.
     |  
     |  partial_fit(self, X, y, classes=None, sample_weight=None)
     |      Perform one epoch of stochastic gradient descent on given samples.
     |      
     |      Internally, this method uses ``max_iter = 1``. Therefore, it is not
     |      guaranteed that a minimum of the cost function is reached after calling
     |      it once. Matters such as objective convergence and early stopping
     |      should be handled by the user.
     |      
     |      Parameters
     |      ----------
     |      X : {array-like, sparse matrix}, shape (n_samples, n_features)
     |          Subset of the training data.
     |      
     |      y : ndarray of shape (n_samples,)
     |          Subset of the target values.
     |      
     |      classes : ndarray of shape (n_classes,), default=None
     |          Classes across all calls to partial_fit.
     |          Can be obtained by via `np.unique(y_all)`, where y_all is the
     |          target vector of the entire dataset.
     |          This argument is required for the first call to partial_fit
     |          and can be omitted in the subsequent calls.
     |          Note that y doesn't need to contain all labels in `classes`.
     |      
     |      sample_weight : array-like, shape (n_samples,), default=None
     |          Weights applied to individual samples.
     |          If not provided, uniform weights are assumed.
     |      
     |      Returns
     |      -------
     |      self :
     |          Returns an instance of self.
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from sklearn.linear_model._stochastic_gradient.BaseSGDClassifier:
     |  
     |  loss_functions = {'epsilon_insensitive': (<class 'sklearn.linear_model...
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from sklearn.linear_model._base.LinearClassifierMixin:
     |  
     |  decision_function(self, X)
     |      Predict confidence scores for samples.
     |      
     |      The confidence score for a sample is the signed distance of that
     |      sample to the hyperplane.
     |      
     |      Parameters
     |      ----------
     |      X : array_like or sparse matrix, shape (n_samples, n_features)
     |          Samples.
     |      
     |      Returns
     |      -------
     |      array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
     |          Confidence scores per (sample, class) combination. In the binary
     |          case, confidence score for self.classes_[1] where >0 means this
     |          class would be predicted.
     |  
     |  predict(self, X)
     |      Predict class labels for samples in X.
     |      
     |      Parameters
     |      ----------
     |      X : array_like or sparse matrix, shape (n_samples, n_features)
     |          Samples.
     |      
     |      Returns
     |      -------
     |      C : array, shape [n_samples]
     |          Predicted class label per sample.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from sklearn.base.ClassifierMixin:
     |  
     |  score(self, X, y, sample_weight=None)
     |      Return the mean accuracy on the given test data and labels.
     |      
     |      In multi-label classification, this is the subset accuracy
     |      which is a harsh metric since you require for each sample that
     |      each label set be correctly predicted.
     |      
     |      Parameters
     |      ----------
     |      X : array-like of shape (n_samples, n_features)
     |          Test samples.
     |      
     |      y : array-like of shape (n_samples,) or (n_samples, n_outputs)
     |          True labels for X.
     |      
     |      sample_weight : array-like of shape (n_samples,), default=None
     |          Sample weights.
     |      
     |      Returns
     |      -------
     |      score : float
     |          Mean accuracy of self.predict(X) wrt. y.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from sklearn.base.ClassifierMixin:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from sklearn.linear_model._stochastic_gradient.BaseSGD:
     |  
     |  set_params(self, **kwargs)
     |      Set and validate the parameters of estimator.
     |      
     |      Parameters
     |      ----------
     |      **kwargs : dict
     |          Estimator parameters.
     |      
     |      Returns
     |      -------
     |      self : object
     |          Estimator instance.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from sklearn.linear_model._base.SparseCoefMixin:
     |  
     |  densify(self)
     |      Convert coefficient matrix to dense array format.
     |      
     |      Converts the ``coef_`` member (back) to a numpy.ndarray. This is the
     |      default format of ``coef_`` and is required for fitting, so calling
     |      this method is only required on models that have previously been
     |      sparsified; otherwise, it is a no-op.
     |      
     |      Returns
     |      -------
     |      self
     |          Fitted estimator.
     |  
     |  sparsify(self)
     |      Convert coefficient matrix to sparse format.
     |      
     |      Converts the ``coef_`` member to a scipy.sparse matrix, which for
     |      L1-regularized models can be much more memory- and storage-efficient
     |      than the usual numpy.ndarray representation.
     |      
     |      The ``intercept_`` member is not converted.
     |      
     |      Returns
     |      -------
     |      self
     |          Fitted estimator.
     |      
     |      Notes
     |      -----
     |      For non-sparse models, i.e. when there are not many zeros in ``coef_``,
     |      this may actually *increase* memory usage, so use this method with
     |      care. A rule of thumb is that the number of zero elements, which can
     |      be computed with ``(coef_ == 0).sum()``, must be more than 50% for this
     |      to provide significant benefits.
     |      
     |      After calling this method, further fitting with the partial_fit
     |      method (if any) will not work until you call densify.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from sklearn.base.BaseEstimator:
     |  
     |  __getstate__(self)
     |  
     |  __repr__(self, N_CHAR_MAX=700)
     |      Return repr(self).
     |  
     |  __setstate__(self, state)
     |  
     |  get_params(self, deep=True)
     |      Get parameters for this estimator.
     |      
     |      Parameters
     |      ----------
     |      deep : bool, default=True
     |          If True, will return the parameters for this estimator and
     |          contained subobjects that are estimators.
     |      
     |      Returns
     |      -------
     |      params : mapping of string to any
     |          Parameter names mapped to their values.
    
    


```python
def sigmoid(z):
  return 1.0/(1.0+np.exp(-z))

z = np.arange(-7,7,0.1)
phi_z = sigmoid(z)
plt.plot(z,phi_z)
plt.axvline(0.0,color='k')
plt.ylim(-0.1,1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')

plt.yticks([0.0,0.5,1.0])
ax = plt.gca()
ax.yaxis.grid(True)
plt.tight_layout()
plt.plot(z,phi_z)
plt.show()

```


    
![png](/img/output_20_0.png)
    



```python
def cost_1(z):
  return -np.log(sigmoid(z))
def cost_0(z):
  return -np.log(1-sigmoid(z))

z = np.arange(-10,10,0.1)
phi_z = sigmoid(z)
c1 = [cost_1(x) for x in z]
plt.plot(phi_z, c1, label='J(w) if y=1')
c0 = [cost_0(x) for x in z]
plt.plot(phi_z, c0, linestyle = '--', label = 'J(w) if y=0')
plt.ylim(0.0,5.1)
plt.xlim([0,1])
plt.xlabel('$\phi$(z)')
plt.ylabel('J(w)')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
```


    
![png](/img/output_21_0.png)
    



```python
class LogisticRegressionGD(object):
  def __init__(self, eta=0.05,n_iter=100,random_state=1):
    self.eta = eta
    self.n_iter = n_iter
    self.random_state = random_state
  
  def fit(self,X,y):
    rgen = np.random.RandomState(self.random_state)
    self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
    self.cost_ = []

    for i in range(self.n_iter):
      net_input = self.net_input(X)
      output = self.activation(net_input)
      errors = (y-output)
      self.w_[1:] += self.eta * X.T.dot(errors)
      self.w_[0] += self.eta * errors.sum()
      cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
      self.cost_.append(cost)

    return self

  def net_input(self, X):
    return np.dot(X, self.w_[1:]) + self.w_[0]
  
  def activation(self, z):
    return 1. /(1. + np.exp(-np.clip(z,-250,250)))
  
  def predict(self,X):
    return np.where(self.net_input(X) >= 0.0, 1, 0)

  
```


```python
X_train_01_subset = X_train[(y_train == 0)|(y_train==1)]
y_train_01_subset = y_train[(y_train == 0)|(y_train==1)]
lrgd = LogisticRegressionGD(eta=0.05,n_iter=1000,random_state=1)
lrgd.fit(X_train_01_subset,y_train_01_subset)

plot_decision_regions(X=X_train_01_subset,y=y_train_01_subset,classifier=lrgd)

plt.xlabel('petal length [std]')
plt.ylabel('petal width [std]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show
```




    <function matplotlib.pyplot.show>




    
![png](/img/output_23_1.png)
    



```python
class LogisticRegressionGD(object):
    """경사 하강법을 사용한 로지스틱 회귀 분류기

    매개변수
    ------------
    eta : float
      학습률 (0.0과 1.0 사이)
    n_iter : int
      훈련 데이터셋 반복 횟수
    random_state : int
      가중치 무작위 초기화를 위한 난수 생성기 시드

    속성
    -----------
    w_ : 1d-array
      학습된 가중치
    cost_ : list
      에포크마다 누적된 로지스틱 비용 함수 값

    """
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """훈련 데이터 학습

        매개변수
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          n_samples 개의 샘플과 n_features 개의 특성으로 이루어진 훈련 데이터
        y : array-like, shape = [n_samples]
          타깃값

        반환값
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            
            # 오차 제곱합 대신 로지스틱 비용을 계산합니다.
            cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        """최종 입력 계산"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        """로지스틱 시그모이드 활성화 계산"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """단위 계단 함수를 사용하여 클래스 레이블을 반환합니다"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
        # 다음과 동일합니다.
        # return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
```


```python
X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
lrgd.fit(X_train_01_subset,
         y_train_01_subset)

plot_decision_regions(X=X_train_01_subset, 
                      y=y_train_01_subset,
                      classifier=lrgd)

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
```


    
![png](/img/output_25_0.png)
    



```python
from sklearn.linear_model import LogisticRegression
```


```python
lr = LogisticRegression(solver='liblinear',multi_class='auto',C=100.0,random_state=1)
lr.fit(X_train_std,y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=lr,test_idx=range(105,150))
plt.xlabel('petal length [std]')
plt.ylabel('petal width [std]')
plt.legend(loc='upper left')
plt.tight_layout
plt.show()
```


    
![png](/img/output_27_0.png)
    



```python
lr.predict_proba(X_test_std[:3,:])
```




    array([[3.17983737e-08, 1.44886616e-01, 8.55113353e-01],
           [8.33962295e-01, 1.66037705e-01, 4.55557009e-12],
           [8.48762934e-01, 1.51237066e-01, 4.63166788e-13]])




```python
lr.predict_proba(X_test_std[:3,:]).argmax(axis=1)
```




    array([2, 0, 0])




```python
lr.predict(X_test_std[:3,:])
```




    array([2, 0, 0])




```python
lr.predict(X_test_std[0,:].reshape(1,-1))
```




    array([2])




```python

```


```python
weights, params = [], []
for c in np.arange(-5,5):
  lr = LogisticRegression(solver = 'liblinear', multi_class='auto' , C=10.**c,random_state=1)
  lr.fit(X_train_std, y_train)
  weights.append(lr.coef_[1])
  params.append(10.**c)

weights = np.array(weights)
plt.plot(params, weights[:,0], label='petal length') 
plt.plot(params, weights[:,1], linestyle='--', label='petal width') 
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()
```


    
![png](/img/output_33_0.png)
    



```python
params
```




    [1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]




```python
from sklearn.svm import SVC
svm = SVC(kernel = 'linear' , C = 1.0 , random_state=1)
svm.fit(X_train_std,y_train)
plot_decision_regions(X_combined_std,
                      y_combined,classifier=svm,test_idx=range(105,150))
plt.xlabel('petal length [std]')
plt.ylabel('petal width [std]')
plt.tight_layout()
plt.show()
```


    
![png](/img/output_35_0.png)
    



```python
from sklearn.linear_model import SGDClassifier
ppn = SGDClassifier(loss='perceptron')
lr = SGDClassifier(loss='log')
svm = SGDClassifier(loss='hinge')
```


```python
np.random.seed(1)
X_xor = np.random.randn(200,2)
y_xor = np.logical_xor(X_xor[:,0]>0,
                       X_xor[:,1]>0)
y_xor = np.where(y_xor,1,-1)
plt.scatter(X_xor[y_xor==1,0],X_xor[y_xor==1,1],c='b',marker='x',label='1')
plt.scatter(X_xor[y_xor==-1,0],X_xor[y_xor==-1,1],c='r',marker='s',label='-1')
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.legend(loc='best')
plt.tight_layout()
plt.show()
```


    
![png](/img/output_37_0.png)
    



```python
svm = SVC(kernel='rbf',random_state=1,gamma=.2,C=10.0)
svm.fit(X_xor,y_xor)
plot_decision_regions(X_xor,y_xor,classifier=svm)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
```


    
![png](/img/output_38_0.png)
    



```python
svm =SVC(kernel ='rbf',random_state=1,gamma=.2,C=1.)
svm.fit(X_train_std,y_train)
plot_decision_regions(X_combined_std,
                      y_combined,
                      classifier=svm,
                      test_idx=range(105,150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
```


    
![png](/img/output_39_0.png)
    



```python
svm =SVC(kernel ='rbf',random_state=1,gamma=100,C=1.)
svm.fit(X_train_std,y_train)
plot_decision_regions(X_combined_std,
                      y_combined,
                      classifier=svm,
                      test_idx=range(105,150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
```


    
![png](/img/output_40_0.png)
    



```python
svm =SVC(kernel ='rbf',random_state=1,gamma=.2,C=100000.)
svm.fit(X_train_std,y_train)
plot_decision_regions(X_combined_std,
                      y_combined,
                      classifier=svm,
                      test_idx=range(105,150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
```


    
![png](/img/output_41_0.png)
    



```python
svm =SVC(kernel ='rbf',random_state=1,gamma=200,C=.1)
svm.fit(X_train_std,y_train)
plot_decision_regions(X_combined_std,
                      y_combined,
                      classifier=svm,
                      test_idx=range(105,150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
```


    
![png](/img/output_42_0.png)
    



```python
svm.coef_
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-42-ce454b79ba3f> in <module>()
    ----> 1 svm.coef_
    

    /usr/local/lib/python3.6/dist-packages/sklearn/svm/_base.py in coef_(self)
        471     def coef_(self):
        472         if self.kernel != 'linear':
    --> 473             raise AttributeError('coef_ is only available when using a '
        474                                  'linear kernel')
        475 
    

    AttributeError: coef_ is only available when using a linear kernel



```python
svm.intercept_
```




    array([-0.35433655, -0.38576202, -0.03143144])




```python
import matplotlib.pyplot as plt
import numpy as np

def gini(p):
  return (p)*(1-(p))+(1-p)*(1-(1-p))

def entropy(p):
  return - p*np.log2(p) - (1-p)*np.log2((1-p))

def error(p):
  return 1 - np.max([p,1-p])

x = np.arange(0.0,1.0,0.01)
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]
err = [error(i) for i in x]
fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls,  c, in zip([ent,sc_ent,gini(x),err],
                           ['Entropy','Entropy (scaled)','Gini Impurity','Misclassification Error'],
                           ['-','-','--','-.'],
                           ['black','lightgray','red','green','cyan']):
  line = ax.plot(x, i, label=lab,
                 linestyle = ls , lw =2 ,color=c)

ax.legend(loc='upper center',bbox_to_anchor=(0.5,1.15), ncol=5, fancybox=True,shadow=False)
ax.axhline(y=0.5,linewidth=1,color='k',linestyle='--')
ax.axhline(y=1,linewidth=1,color='k',linestyle='--')
plt.ylim([0,1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.show()
```


    
![png](/img/output_45_0.png)
    



```python
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=1)
tree.fit(X_train,y_train)
X_combined = np.vstack((X_train,X_test))
y_combined = np.hstack((y_train,y_test))
plot_decision_regions(X_combined, y_combined, classifier=tree,
                      test_idx = range(105,150))

plt.xlabel('petal length')
plt.ylabel('petal width')
plt.tight_layout()
plt.show()
```


    
![png](/img/output_46_0.png)
    



```python
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz

```


```python
dot_data = export_graphviz(tree,filled= True,
                           rounded=True,
                           class_names = ['Setosa','Versicolor','Virginica'],
                           feature_names=['petal length','petal width'],
                           out_file = None)

graph = graph_from_dot_data(dot_data)
graph.write_png('/content/gdrive/MyDrive/Colab Notebooks/data/tree.png')

```




    True




```python
from sklearn.ensemble import RandomForestClassifier
```


```python
forest = RandomForestClassifier(criterion='gini',
                                n_estimators=25,
                                random_state=1,
                                n_jobs=2)
forest.fit(X_train,y_train)
plot_decision_regions(X_combined,y_combined,classifier=forest,test_idx=range(105,150))
plt.show()
```


    
![png](/img/output_50_0.png)
    



```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3 , p = 2, metric = 'minkowski')
knn.fit(X_train_std,y_train)
plot_decision_regions(X_combined_std,y_combined,classifier = knn, test_idx=range(105,150))
plt.show()
```


    
![png](/img/output_51_0.png)
    



```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 10 , p = 2, metric = 'minkowski')
knn.fit(X_train_std,y_train)
plot_decision_regions(X_combined_std,y_combined,classifier = knn, test_idx=range(105,150))
plt.show()
```


    
![png](/img/output_52_0.png)
    



```python

```
