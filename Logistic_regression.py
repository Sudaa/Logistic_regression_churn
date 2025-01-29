### import librabries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np

## Load dataset
df = pd.read_csv("ChurnData.csv")

## data pre-processing and selection 
## we can select specific column or features for modeling 
churn_df = df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless', 'churn']]
churn_df['churn'] = churn_df['churn'].astype('int')

churn_df.shape

## define X (features) and Y (target) varibales for our dataset
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])

y = np.asarray(churn_df['churn'])

###normalize the dataset
X = preprocessing.StandardScaler().fit(X).transform(X)

## Train/Test dataset
from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=4)
print('Train set: ', X_train.shape, y_train.shape)
print('Test set: ', X_test.shape, y_test.shape)

##Modeling 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
LR = LogisticRegression(C=0.01, solver= 'liblinear').fit(X_train, y_train)
LR

## predict using test set
yhat = LR.predict(X_test)

## probability of prediction
yhat_prob = LR.predict_proba(X_test)
yhat_prob

## Evaluation , jaccard index
from sklearn.metrics import jaccard_score
jaccard_score(y_test, yhat, pos_label= 0)

## confusion matrix 
from sklearn.metrics import classification_report, confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap = plt.cm.Blues):  ## this function prints and plots the confusion matrix, normalizzation can be applied by setting 'normalization=True'
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

print(confusion_matrix(y_test, yhat, labels=[1,0]))

## compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels= [1,0])
np.set_printoptions(precision=2)

##plot non-normalized confusion matix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1', 'churn=0'], normalize=False, title= 'Confusion matrix')
plt.show()

##classification_report
print(classification_report(y_test, yhat))