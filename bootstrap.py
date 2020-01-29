import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.linear_model
import matplotlib.pylab as plt

data = np.loadtxt("notas_andes.dat", skiprows=1)
Y = data[:,4]
X = data[:,:4]

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.3)
regresion = sklearn.linear_model.LinearRegression()

B=[]
for i in range(1000):
    randind = np.random.randint(len(X),size=69)
    Xran=X[randind]
    Yran=Y[randind]
    regresion.fit(Xran, Yran)
    B.append(regresion.coef_)

B1=np.array(B)
plt.figure(figsize=(18,12))
for i in range(4):
    plt.subplot(2,2,i+1)
    m=np.mean(B1[:,i])
    sigma=np.std(B1[:,i])
    plt.hist(B1[:,i],bins=60,label='{:.2f} $\pm$ {:.2f}'.format(m,sigma))
    plt.legend(loc=0.0)
    plt.vlines(m, 0, 55)
    plt.title('Beta '+str(i+1))
plt.savefig('bootstrap.png')