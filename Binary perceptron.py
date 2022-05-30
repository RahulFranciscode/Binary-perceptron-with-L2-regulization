import numpy as np
import pandas as pd


class Perceptron:
    def __init__(self, max_iters=20):
        self.max_iters = max_iters
        self.activation_function = self._unit_step_function
        self.activation_func=self.unit_step_func
        self.w = None
        self.b = None

    def fit(self, X, y):
        row,column = X.shape

        # init parameters
        self.w = np.zeros(column)
        self.b = 0
        X = X.astype(np.float64)
        y = y.astype(np.float64)

        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.max_iters):

            for idx, x_i in enumerate(X):
                

                linearoutput = np.dot(x_i, self.w) + self.b
                y_predicted = self.activation_function(linearoutput)
               
                # Perceptron update rule
                new_update = y_[idx] - y_predicted

                self.w = self.w + new_update * x_i
                self.b = self.b+ new_update
        return self

    def predict(self, X):
        linearoutput = np.dot(X, self.w) + self.b
        y_predicted = self.activation_function(linearoutput)
        return y_predicted
        #return linearoutput

    def _unit_step_function(self, x):
        return np.where(x >= 0, 1, 0)
    def Regularisation_training(self, X,y, lamba):
            
            self.lamba=lamba
            row,column = X.shape

            self.w = np.zeros(column)
            self.b = 0
            y_ = np.array([1 if i > 0 else 0 for i in y])

            for _ in range(self.max_iters):

                for idx, x_i in enumerate(X):

                    linearoutput1= np.dot(x_i, self.w) + self.b
                    y_predicted = self.activation_func(linearoutput1)
               
                    # Perceptron update rule
                    d = y_[idx] - y_predicted
                    self.w = (1 - (2 * self.lamba)) * self.w + (d * x_i) 
                    self.b = self.b + d  
            return self
    def prediction(self, X):
        linearoutput1 = np.dot(X, self.w) + self.b
        return linearoutput1
    def unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)
    
def readfile(filename):
        jay = []
        with open(filename) as file:
            for page in file:
                jay.append(page.rstrip().split(","))
            arr = np.array(jay)
            w1 = arr[arr[:, 4] == "class-1"]
            w2 = arr[arr[:, 4] == "class-2"]
            w3 = arr[arr[:, 4] == "class-3"]
        return w1, w2, w3
    
trainingclass1,trainingclass2, trainingclass3 = readfile("train.data.csv")
testingclass1, testingclass2, testingclass3 = readfile("test.data.csv")
# applying binary perceptron 
print("----------------------------------------------------------------------")
print("Q3. Binary Perceptron Accuracy\n----------------------------------------------------------------------")


# a) Class 1 vs Class 2
PERCEPTRON1 = Perceptron()
class1v2 = np.concatenate((trainingclass1, trainingclass2), axis=0)
X1=class1v2[:,0:4]

X1=X1.astype(np.float64)
y1=class1v2[:,4]

y1=np.where(class1v2[:,4]=='class-1',1,0)
y1=y1.astype(np.float64)
PERCEPTRON1.fit(X1,y1)
#class 1 v class 2 test
class1v2T = np.concatenate((testingclass1, testingclass2), axis=0)
X1T=class1v2T[:,0:4]
X1T=X1T.astype(np.float64)
y1T=np.where(class1v2T[:,4]=='class-1',1,0)
y1T=y1T.astype(np.float64)
predictions = PERCEPTRON1.predict(X1)

predictions_T =PERCEPTRON1.predict(X1T)
accuracy = np.sum(y1 == predictions) / len(y1)*100
accuracy_t = np.sum(y1T == predictions_T) / len(y1T)*100
print("Train Accuracy of Classification Class-1 and Class-2= "+ str(accuracy_t))
print("Test Accuracy of Classification Class-1 and Class-2 = "+ str(accuracy))
print("*******************")

#  Class 2 vs Class 3
PERCEPTRON2 = Perceptron()
class2v3 = np.concatenate((trainingclass2, trainingclass3), axis=0)
X2=class2v3[:,0:4]

X2=X2.astype(np.float64)
y2=class2v3[:,4]

y2=np.where(class2v3[:,4]=='class-2',1,0)
y2=y2.astype(np.float64)
PERCEPTRON2.fit(X2,y2)
#class 2 v class 3 test
class2v3T = np.concatenate((testingclass2, testingclass3), axis=0)
X2T=class2v3T[:,0:4]
X2T=X2T.astype(np.float64)
y2T=np.where(class2v3T[:,4]=='class-2',1,0)
y2T=y2T.astype(np.float64)
predictionss= PERCEPTRON2.predict(X2)

predictions_TT=PERCEPTRON2.predict(X2T)
accuracy = np.sum(y2 == predictionss)/ len(y2)*100
accuracy_t = np.sum(y2T == predictions_TT)/ len(y2T)*100
print("Train Accuracy of Classification Class-2 nd Class-3= "+ str(accuracy_t))
print("Test Accuracy of Classification Class-2 and Class-3 = "+ str(accuracy))
print("*******************")
# a) Class 1 vs Class 3
PERCEPTRON3= Perceptron()
class1v3= np.concatenate((trainingclass1, trainingclass3), axis=0)
X3=class1v3[:,0:4]

X3=X3.astype(np.float64)
y3=class1v3[:,4]

y3=np.where(class1v3[:,4]=='class-3',1,0)
y3=y3.astype(np.float64)
PERCEPTRON3.fit(X3,y3)
#class 1 v class 3 test
class1v3T = np.concatenate((testingclass1, testingclass3), axis=0)
X3T=class1v3T[:,0:4]
X3T=X3T.astype(np.float64)
y3T=np.where(class1v3T[:,4]=='class-3',1,0)
y3T=y3T.astype(np.float64)
predictions = PERCEPTRON3.predict(X3)

predictions_T =PERCEPTRON3.predict(X3T)
accuracy = np.sum(y3 == predictions) / len(y3)*100
accuracy_t = np.sum(y3T == predictions_T) / len(y3T)*100
print("Train Accuracy of Classification Class-1 and Class-3= "+ str(accuracy_t))
print("Test Accuracy of Classification Class-1 and Class-3 = "+ str(accuracy))



print("-------------------------------------------------------------------------")

print("Q3. multiclass perceptron accuracy\n-----------------------------------------------------------------")

#Q4.  multi-class classification using the 1-vs-rest approach.
#-------------------class1---------------------------------------------------------
def multi_classifier():
    class1v2 = np.concatenate((trainingclass1, trainingclass2), axis=0)
    class1v2 = np.concatenate((class1v2, trainingclass3), axis=0)

    class1v2T = np.concatenate((testingclass1, testingclass2), axis=0)
    class1v2T = np.concatenate((class1v2T, testingclass3), axis=0)
  
    w1= np.where(class1v2[:,4] == "class-1", 1, -1)

    w1=w1.astype(np.float64)



    p_multi_class1 = Perceptron()

    p_multi_class1.fit(class1v2[:,0:4],w1)

    predictions_m1=p_multi_class1.predict(class1v2T[:,:4].astype(np.float64))


    predictions_m1_t =p_multi_class1.predict(class1v2[:,:4].astype(np.float64))


 #--------- class 2 -------------------------------------------
    w2= np.where(class1v2[:,4] == "class-2", 1, -1)

    w2=w2.astype(np.float64)



    p_multi_class2 = Perceptron()

    p_multi_class2.fit(class1v2[:,0:4],w2)

    predictions_m2=p_multi_class2.predict(class1v2T[:,:4].astype(np.float64))


    predictions_m2_t =p_multi_class2.predict(class1v2[:,:4].astype(np.float64))

 #--------- class 3 -------------------------------------------
    w3= np.where(class1v2[:,4] == "class-3", 1, -1)

    w3=w3.astype(np.float64)



    p_multi_class3 = Perceptron()

    p_multi_class3.fit(class1v2[:,0:4],w3)

    predictions_m3=p_multi_class3.predict(class1v2T[:,:4].astype(np.float64))


    predictions_m3_t =p_multi_class3.predict(class1v2[:,:4].astype(np.float64))
    return predictions_m1, predictions_m2, predictions_m3,predictions_m1_t,predictions_m2_t,predictions_m3_t

def compare_accuries(pred1,pred2,pred3,classes):
    return classes[np.argmax((pred1,pred2,pred3),0)]

class_labels = np.array(['class-%d'%(i+1) for i in range(3)])
predictions_m1, predictions_m2, predictions_m3,predictions_m1_t,predictions_m2_t,predictions_m3_t= multi_classifier()

result_multi=compare_accuries(predictions_m1,predictions_m2,predictions_m3,class_labels)
result_multi_t=compare_accuries(predictions_m2_t,predictions_m2_t,predictions_m3_t,class_labels)
class1v2T = np.concatenate((testingclass1, testingclass2), axis=0)
class1v2T = np.concatenate((class1v2T, testingclass3), axis=0)



multi_class_accuracy = np.sum(class1v2T[:,4] == result_multi) /len(class1v2T[:,4]) * 100
print("test data set:",multi_class_accuracy)

class1v2 = np.concatenate((trainingclass1, trainingclass2), axis=0)
class1v2 = np.concatenate((class1v2, trainingclass3), axis=0)

multi_class_accuracy_t = np.sum(class1v2[:,4] == result_multi_t) /len(class1v2[:,4]) * 100
print("train data set :",multi_class_accuracy_t)
print("------------------------------------------------------------------------------")


print("Q3. multiclass perceptron accuracy for each lamda value regularization\n------------------------------------------------------------------------------\n")

# Lamda 0.01
def multi_classifier_reg():
    class1v2 = np.concatenate((trainingclass1, trainingclass2), axis=0)
    class1v2 = np.concatenate((class1v2, trainingclass3), axis=0)

    class1v2T = np.concatenate((testingclass1, testingclass2), axis=0)
    class1v2T = np.concatenate((class1v2T, testingclass3), axis=0)
   
    w1= np.where(class1v2[:,4] == "class-1", 1, -1)

    w1=w1.astype(np.float64)



    p_multi_class1 = Perceptron()

    p_multi_class1.Regularisation_training(class1v2[:,0:4].astype(np.float64),w1,0.01)
    predictions_mm1=p_multi_class1.predict(class1v2T[:,:4].astype(np.float64))


    predictions_mm1_t =p_multi_class1.prediction(class1v2[:,:4].astype(np.float64))
       #--------- class 2 -------------------------------------------
    w2= np.where(class1v2[:,4] == "class-1", 1, -1)

    w2=w2.astype(np.float64)



    p_multi_class2 = Perceptron()

    p_multi_class2.Regularisation_training(class1v2[:,0:4].astype(np.float64),w2,0.01)

    predictions_mm2=p_multi_class2.predict(class1v2T[:,:4].astype(np.float64))


    predictions_mm2_t =p_multi_class2.prediction(class1v2[:,:4].astype(np.float64))
     #--------- class 3-------------------------------------------
    w3= np.where(class1v2[:,4] == "class-1", 1, -1)

    w3=w3.astype(np.float64)



    p_multi_class3 = Perceptron()

    p_multi_class3.Regularisation_training(class1v2[:,0:4].astype(np.float64),w3,0.01)

    predictions_mm3=p_multi_class3.predict(class1v2T[:,:4].astype(np.float64))


    predictions_mm3_t =p_multi_class3.prediction(class1v2[:,:4].astype(np.float64))
    return  predictions_mm1,predictions_mm2,predictions_mm3,predictions_mm1_t,predictions_mm2_t, predictions_mm3_t

def compareaccuries(pred1,pred2,pred3,classes):
    return classes[np.argmax((pred1,pred2,pred3),0)]

class_labels = np.array(['class-%d'%(i+1) for i in range(3)])
predictions_mm1,predictions_mm2,predictions_mm3,predictions_mm1_t,predictions_mm2_t, predictions_mm3_t= multi_classifier_reg()

result_multi=compare_accuries(predictions_mm1,predictions_mm2,predictions_mm3,class_labels)
result_multi_t=compare_accuries(predictions_mm1_t,predictions_mm2_t, predictions_mm3_t,class_labels)
class1v2T = np.concatenate((testingclass1, testingclass2), axis=0)
class1v2T = np.concatenate((class1v2T, testingclass3), axis=0)


multi_class_accuracy = np.sum(class1v2T[:,4] == result_multi) /len(class1v2T[:,4]) * 100
print("test for lamda 0.01",multi_class_accuracy)

class1v2 = np.concatenate((trainingclass1, trainingclass2), axis=0)
class1v2 = np.concatenate((class1v2, trainingclass3), axis=0)
multi_class_accuracy_t = np.sum(class1v2[:,4] == result_multi_t) /len(class1v2[:,4]) * 100
print("ttraining fr lamda 0.01", multi_class_accuracy_t)
print("----------------------------------------------------------------------------------------------")
#lamda .1
def multi_classifier_regg():
    class1v2 = np.concatenate((trainingclass1, trainingclass2), axis=0)
    class1v2 = np.concatenate((class1v2, trainingclass3), axis=0)

    class1v2T = np.concatenate((testingclass1, testingclass2), axis=0)
    class1v2T = np.concatenate((class1v2T, testingclass3), axis=0)
   
    w1= np.where(class1v2[:,4] == "class-1", 1, -1)

    w1=w1.astype(np.float64)



    p_multi_class1 = Perceptron()

    p_multi_class1.Regularisation_training(class1v2[:,0:4].astype(np.float64),w1,0.1)
    predictions_mm1=p_multi_class1.predict(class1v2T[:,:4].astype(np.float64))


    predictions_mm1_t =p_multi_class1.prediction(class1v2[:,:4].astype(np.float64))
       #--------- class 2 -------------------------------------------
    w2= np.where(class1v2[:,4] == "class-1", 1, -1)

    w2=w2.astype(np.float64)



    p_multi_class2 = Perceptron()

    p_multi_class2.Regularisation_training(class1v2[:,0:4].astype(np.float64),w2,0.1)

    predictions_mm2=p_multi_class2.predict(class1v2T[:,:4].astype(np.float64))


    predictions_mm2_t =p_multi_class2.prediction(class1v2[:,:4].astype(np.float64))
     #--------- class 3-------------------------------------------
    w3= np.where(class1v2[:,4] == "class-1", 1, -1)

    w3=w3.astype(np.float64)



    p_multi_class3 = Perceptron()

    p_multi_class3.Regularisation_training(class1v2[:,0:4].astype(np.float64),w3,0.1)

    predictions_mm3=p_multi_class3.predict(class1v2T[:,:4].astype(np.float64))


    predictions_mm3_t =p_multi_class3.prediction(class1v2[:,:4].astype(np.float64))
    return  predictions_mm1,predictions_mm2,predictions_mm3,predictions_mm1_t,predictions_mm2_t, predictions_mm3_t

def compareaccuries(pred1,pred2,pred3,classes):
    return classes[np.argmax((pred1,pred2,pred3),0)]

class_labels = np.array(['class-%d'%(i+1) for i in range(3)])
predictions_mm1,predictions_mm2,predictions_mm3,predictions_mm1_t,predictions_mm2_t, predictions_mm3_t= multi_classifier_regg()

result_multi=compare_accuries(predictions_mm1,predictions_mm2,predictions_mm3,class_labels)
result_multi_t=compare_accuries(predictions_mm1_t,predictions_mm2_t, predictions_mm3_t,class_labels)
class1v2T = np.concatenate((testingclass1, testingclass2), axis=0)
class1v2T = np.concatenate((class1v2T, testingclass3), axis=0)


multi_class_accuracy = np.sum(class1v2T[:,4] == result_multi) /len(class1v2T[:,4]) * 100
print("test for lamda for .1:",multi_class_accuracy)

class1v2 = np.concatenate((trainingclass1, trainingclass2), axis=0)
class1v2 = np.concatenate((class1v2, trainingclass3), axis=0)
multi_class_accuracy_t = np.sum(class1v2[:,4] == result_multi_t) /len(class1v2[:,4]) * 100
print("train fr lamda for .1:",multi_class_accuracy_t)
print("----------------------------------------------------------------------------------")
# lamda 1
def multi_classifier_reg():
    class1v2 = np.concatenate((trainingclass1, trainingclass2), axis=0)
    class1v2 = np.concatenate((class1v2, trainingclass3), axis=0)

    class1v2T = np.concatenate((testingclass1, testingclass2), axis=0)
    class1v2T = np.concatenate((class1v2T, testingclass3), axis=0)
   
    w1= np.where(class1v2[:,4] == "class-1", 1, -1)

    w1=w1.astype(np.float64)



    p_multi_class1 = Perceptron()

    p_multi_class1.Regularisation_training(class1v2[:,0:4].astype(np.float64),w1,1)
    predictions_mm1=p_multi_class1.predict(class1v2T[:,:4].astype(np.float64))


    predictions_mm1_t =p_multi_class1.prediction(class1v2[:,:4].astype(np.float64))
       #--------- class 2 -------------------------------------------
    w2= np.where(class1v2[:,4] == "class-1", 1, -1)

    w2=w2.astype(np.float64)



    p_multi_class2 = Perceptron()

    p_multi_class2.Regularisation_training(class1v2[:,0:4].astype(np.float64),w2,1)

    predictions_mm2=p_multi_class2.predict(class1v2T[:,:4].astype(np.float64))


    predictions_mm2_t =p_multi_class2.prediction(class1v2[:,:4].astype(np.float64))
     #--------- class 3-------------------------------------------
    w3= np.where(class1v2[:,4] == "class-1", 1, -1)

    w3=w3.astype(np.float64)



    p_multi_class3 = Perceptron()

    p_multi_class3.Regularisation_training(class1v2[:,0:4].astype(np.float64),w3,1)

    predictions_mm3=p_multi_class3.predict(class1v2T[:,:4].astype(np.float64))


    predictions_mm3_t =p_multi_class3.prediction(class1v2[:,:4].astype(np.float64))
    return  predictions_mm1,predictions_mm2,predictions_mm3,predictions_mm1_t,predictions_mm2_t, predictions_mm3_t

def compareaccuries(pred1,pred2,pred3,classes):
    return classes[np.argmax((pred1,pred2,pred3),0)]

class_labels = np.array(['class-%d'%(i+1) for i in range(3)])
predictions_mm1,predictions_mm2,predictions_mm3,predictions_mm1_t,predictions_mm2_t, predictions_mm3_t= multi_classifier_reg()

result_multi=compare_accuries(predictions_mm1,predictions_mm2,predictions_mm3,class_labels)
result_multi_t=compare_accuries(predictions_mm1_t,predictions_mm2_t, predictions_mm3_t,class_labels)
class1v2T = np.concatenate((testingclass1, testingclass2), axis=0)
class1v2T = np.concatenate((class1v2T, testingclass3), axis=0)


multi_class_accuracy = np.sum(class1v2T[:,4] == result_multi) /len(class1v2T[:,4]) * 100
print("test for lamda 1:",multi_class_accuracy)

class1v2 = np.concatenate((trainingclass1, trainingclass2), axis=0)
class1v2 = np.concatenate((class1v2, trainingclass3), axis=0)
multi_class_accuracy_t = np.sum(class1v2[:,4] == result_multi_t) /len(class1v2[:,4]) * 100
print("train fr lamda 1:",multi_class_accuracy_t)
print("----------------------------------------------------------------------------------------")
# lamda 10
def multi_classifier_regg():
    class1v2 = np.concatenate((trainingclass1, trainingclass2), axis=0)
    class1v2 = np.concatenate((class1v2, trainingclass3), axis=0)

    class1v2T = np.concatenate((testingclass1, testingclass2), axis=0)
    class1v2T = np.concatenate((class1v2T, testingclass3), axis=0)
   
    w1= np.where(class1v2[:,4] == "class-1", 1, -1)

    w1=w1.astype(np.float64)



    p_multi_class1 = Perceptron()

    p_multi_class1.Regularisation_training(class1v2[:,0:4].astype(np.float64),w1,10)
    predictions_mm1=p_multi_class1.predict(class1v2T[:,:4].astype(np.float64))


    predictions_mm1_t =p_multi_class1.prediction(class1v2[:,:4].astype(np.float64))
       #--------- class 2 -------------------------------------------
    w2= np.where(class1v2[:,4] == "class-1", 1, -1)

    w2=w2.astype(np.float64)



    p_multi_class2 = Perceptron()

    p_multi_class2.Regularisation_training(class1v2[:,0:4].astype(np.float64),w2,10)

    predictions_mm2=p_multi_class2.predict(class1v2T[:,:4].astype(np.float64))


    predictions_mm2_t =p_multi_class2.prediction(class1v2[:,:4].astype(np.float64))
     #--------- class 3-------------------------------------------
    w3= np.where(class1v2[:,4] == "class-1", 1, -1)

    w3=w3.astype(np.float64)



    p_multi_class3 = Perceptron()

    p_multi_class3.Regularisation_training(class1v2[:,0:4].astype(np.float64),w3,10)

    predictions_mm3=p_multi_class3.predict(class1v2T[:,:4].astype(np.float64))


    predictions_mm3_t =p_multi_class3.prediction(class1v2[:,:4].astype(np.float64))
    return  predictions_mm1,predictions_mm2,predictions_mm3,predictions_mm1_t,predictions_mm2_t, predictions_mm3_t

def compareaccuries(pred1,pred2,pred3,classes):
    return classes[np.argmax((pred1,pred2,pred3),0)]

class_labels = np.array(['class-%d'%(i+1) for i in range(3)])
predictions_mm1,predictions_mm2,predictions_mm3,predictions_mm1_t,predictions_mm2_t, predictions_mm3_t= multi_classifier_regg()

result_multi=compare_accuries(predictions_mm1,predictions_mm2,predictions_mm3,class_labels)
result_multi_t=compare_accuries(predictions_mm1_t,predictions_mm2_t, predictions_mm3_t,class_labels)
class1v2T = np.concatenate((testingclass1, testingclass2), axis=0)
class1v2T = np.concatenate((class1v2T, testingclass3), axis=0)


multi_class_accuracy = np.sum(class1v2T[:,4] == result_multi) /len(class1v2T[:,4]) * 100
print("test for lamda 10:",multi_class_accuracy)

class1v2 = np.concatenate((trainingclass1, trainingclass2), axis=0)
class1v2 = np.concatenate((class1v2, trainingclass3), axis=0)
multi_class_accuracy_t = np.sum(class1v2[:,4] == result_multi_t) /len(class1v2[:,4]) * 100
print("train fr lamda 10:",multi_class_accuracy_t)
print("---------------------------------------------------------------------")

# lamda 100
def multi_classifier_regg():
    class1v2 = np.concatenate((trainingclass1, trainingclass2), axis=0)
    class1v2 = np.concatenate((class1v2, trainingclass3), axis=0)

    class1v2T = np.concatenate((testingclass1, testingclass2), axis=0)
    class1v2T = np.concatenate((class1v2T, testingclass3), axis=0)
   
    w1= np.where(class1v2[:,4] == "class-1", 1, -1)

    w1=w1.astype(np.float64)



    p_multi_class1 = Perceptron()

    p_multi_class1.Regularisation_training(class1v2[:,0:4].astype(np.float64),w1,100)
    predictions_mm1=p_multi_class1.predict(class1v2T[:,:4].astype(np.float64))


    predictions_mm1_t =p_multi_class1.prediction(class1v2[:,:4].astype(np.float64))
       #--------- class 2 -------------------------------------------
    w2= np.where(class1v2[:,4] == "class-1", 1, -1)

    w2=w2.astype(np.float64)



    p_multi_class2 = Perceptron()

    p_multi_class2.Regularisation_training(class1v2[:,0:4].astype(np.float64),w2,100)

    predictions_mm2=p_multi_class2.predict(class1v2T[:,:4].astype(np.float64))


    predictions_mm2_t =p_multi_class2.prediction(class1v2[:,:4].astype(np.float64))
     #--------- class 3-------------------------------------------
    w3= np.where(class1v2[:,4] == "class-1", 1, -1)

    w3=w3.astype(np.float64)



    p_multi_class3 = Perceptron()

    p_multi_class3.Regularisation_training(class1v2[:,0:4].astype(np.float64),w3,100)

    predictions_mm3=p_multi_class3.predict(class1v2T[:,:4].astype(np.float64))


    predictions_mm3_t =p_multi_class3.prediction(class1v2[:,:4].astype(np.float64))
    return  predictions_mm1,predictions_mm2,predictions_mm3,predictions_mm1_t,predictions_mm2_t, predictions_mm3_t

def compareaccuries(pred1,pred2,pred3,classes):
    return classes[np.argmax((pred1,pred2,pred3),0)]

class_labels = np.array(['class-%d'%(i+1) for i in range(3)])
predictions_mm1,predictions_mm2,predictions_mm3,predictions_mm1_t,predictions_mm2_t, predictions_mm3_t= multi_classifier_regg()

result_multi=compare_accuries(predictions_mm1,predictions_mm2,predictions_mm3,class_labels)
result_multi_t=compare_accuries(predictions_mm1_t,predictions_mm2_t, predictions_mm3_t,class_labels)
class1v2T = np.concatenate((testingclass1, testingclass2), axis=0)
class1v2T = np.concatenate((class1v2T, testingclass3), axis=0)


multi_class_accuracy = np.sum(class1v2T[:,4] == result_multi) /len(class1v2T[:,4]) * 100
print("test for lamda 100:",multi_class_accuracy)

class1v2 = np.concatenate((trainingclass1, trainingclass2), axis=0)
class1v2 = np.concatenate((class1v2, trainingclass3), axis=0)
multi_class_accuracy_t = np.sum(class1v2[:,4] == result_multi_t) /len(class1v2[:,4]) * 100
print("train fr lamda 100:",multi_class_accuracy_t)


