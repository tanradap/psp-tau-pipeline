# Functions for custom classification metrics 

from sklearn.metrics import confusion_matrix


# Accuracy per class: tau & Non-tau


def NT_acc(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,
                          y_pred,
                          labels=['Non_tau', 'Tau'],
                          normalize='true'
                          )
    acc_c = cm.diagonal()
    return acc_c[0]


def T_acc(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,
                          y_pred,
                          labels=['Non_tau','Tau'],
                          normalize='true'
                          )
    acc_c = cm.diagonal()
    return acc_c[1]


def CB_acc(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,
                          y_pred,
                          labels=['CB','NFT','Others','TA'],
                          normalize='true'
                          )
    acc_c = cm.diagonal()
    return acc_c[0]


def NFT_acc(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,
                          y_pred,
                          labels=['CB','NFT','Others','TA'],
                          normalize='true'
                          )
    acc_c = cm.diagonal()
    return acc_c[1]


def Others_acc(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,
                          y_pred,
                          labels=['CB','NFT','Others','TA'],
                          normalize='true'
                          )
    acc_c = cm.diagonal()
    return acc_c[2] 


def TA_acc(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,
                          y_pred,
                          labels=['CB','NFT','Others','TA'],
                          normalize='true'
                          )
    acc_c = cm.diagonal()
    return acc_c[3] 


def CB_acc_noTA(clf, X, y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,
                          y_pred,
                          labels=['CB','NFT','Others'],
                          normalize='true'
                          )
    acc_c = cm.diagonal()
    return acc_c[0]


def NFT_acc_noTA(clf, X, y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,
                          y_pred,
                          labels=['CB', 'NFT', 'Others'],
                          normalize='true'
                          )
    acc_c = cm.diagonal()
    return acc_c[1] 


def Others_acc_noTA(clf, X, y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,
                          y_pred,
                          labels=['CB', 'NFT', 'Others'],
                          normalize='true'
                          )
    acc_c = cm.diagonal()
    return acc_c[2] 


# Confusion per class:


def NT_as_T(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,
                          y_pred,
                          labels=['Non_tau', 'Tau'],
                          normalize='true'
                          )
    return cm[0][1]


def T_as_NT(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,
                          y_pred,
                          labels=['Non_tau', 'Tau'],
                          normalize='true'
                          )
    return cm[1][0]


def CB_as_NFT(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,
                          y_pred,
                          labels=['CB', 'NFT', 'Others', 'TA'],
                          normalize='true'
                          )
    return cm[0][1]


def CB_as_Others(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,
                          y_pred,
                          labels=['CB', 'NFT', 'Others', 'TA'],
                          normalize='true'
                          )
    return cm[0][2]  


def CB_as_TA(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,
                          y_pred,
                          labels=['CB', 'NFT', 'Others', 'TA'],
                          normalize='true'
                          )
    return cm[0][3]


def NFT_as_CB(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,
                          y_pred,
                          labels=['CB', 'NFT', 'Others', 'TA'],
                          normalize='true')
    return cm[1][0]


def NFT_as_Others(clf, X, y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,
                          y_pred,
                          labels=['CB', 'NFT', 'Others', 'TA'],
                          normalize='true')
    return cm[1][2]


def NFT_as_TA(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,
                          y_pred,
                          labels=['CB', 'NFT', 'Others', 'TA'],
                          normalize='true')
    return cm[1][3]


def Others_as_CB(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,
                          y_pred,
                          labels=['CB', 'NFT', 'Others', 'TA'],
                          normalize='true')
    return cm[2][0]


def Others_as_NFT(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,
                          y_pred,
                          labels=['CB', 'NFT', 'Others', 'TA'],
                          normalize='true')
    return cm[2][1]


def Others_as_TA(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,
                          y_pred,
                          labels=['CB', 'NFT', 'Others', 'TA'],
                          normalize='true')
    return cm[2][3]


def TA_as_CB(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,
                          y_pred,
                          labels=['CB', 'NFT', 'Others', 'TA'],
                          normalize='true')
    return cm[3][0]


def TA_as_NFT(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,
                          y_pred,
                          labels=['CB', 'NFT', 'Others', 'TA'],
                          normalize='true')
    return cm[3][1]


def TA_as_Others(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,
                          y_pred,
                          labels=['CB', 'NFT', 'Others', 'TA'],
                          normalize='true')
    return cm[3][2]
