import math
import numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas

def lectura(doc):
   
    data = []
    lines = []
    count = 0
    with open(doc) as f:
        lines = f.readlines()
    
    for line in lines:
        data.append(line)
        data[count]= line.split(",")        
        count += 1
    return data

def normalizarBanco(data):
    cont = 0
    cont2 = 0
    minBal = float(data[1][3])
    maxBal = float(data[1][3])
    minIn= float(data[1][4]) 
    maxIn = float(data[1][4]) 

    data.pop(0)

    i = 0
    while i < len(data):
        data[i].pop(0)
        i +=1

    while(len(data) > cont):
        if( data[cont][0] == "No"):
           data[cont][0] = 0 
        else:
            data[cont][0] = 1 

        if( data[cont][1] == "No"):
           data[cont][1] = 0 
        else:
            data[cont][1] = 1 

        if(minBal > float(data[cont][2])):
            minBal =  float(data[cont][2])
        if(maxBal <  float(data[cont][2])):
            maxBal =  float(data[cont][2])

        if(minIn >  float(data[cont][3])):
            minIn =  float(data[cont][3])
        if(maxIn <  float(data[cont][3])):
            maxIn = float(data[cont][3])
        
        data[cont].append(1)

        cont +=1

    while(len(data) > cont2):
        
        data[cont2][2] = ( float(data[cont2][2]) - float(minBal))/(float(maxBal)-float(minBal))
        data[cont2][3] = ( float(data[cont2][3]) - float(minIn))/(float(maxIn)-float(minIn))

        cont2 +=1
    return data
   

   
def normalizarGenero(data):
    
    data.pop(0)
    cont = 0
    cont2 = 0
    minH = float(data[1][1])
    maxH = float(data[1][1])
    minW = float(data[1][2]) 
    maxW = float(data[1][2])

    while(len(data) > cont):
        if( data[cont][0] == '"Male"'):
           data[cont][0] = 0 
        else:
            data[cont][0] = 1 

        if(minH > float(data[cont][1])):
            minH = float(data[cont][1])
        if(maxH < float(data[cont][1])):
            maxH = float(data[cont][1])

        if(minW > float(data[cont][2])):
            minW = float(data[cont][2])
        if(maxW < float(data[cont][2])):
            maxW = float(data[cont][2])
        
        data[cont].append(1)

        cont +=1
    while(len(data) > cont2):
        
        data[cont2][1] = ( float(data[cont2][1]) - float(minH))/(float(maxH)-float(minH))
        data[cont2][2] = ( float(data[cont2][2]) - float(minW))/(float(maxW)-float(minW))

        cont2 +=1
    return data

def Y(data):
    i = 0  
    y = []

    while i < len(data):
        y.append(data[i][0])
        data[i].pop(0)
        i += 1

    return y

def sigmoide(t):
    return 1/(1 + numpy.exp(-t))

def gradiente(data, alfa, porcenTest, porcenTrain, threshold, semilla):
  
    dataTrain, dataTest,  yTrain, yTest = train_test_split(data, Y(data), train_size=(float(porcenTrain)/100), test_size=(float(porcenTest)/100),random_state=int(semilla))
  
    dataTrain = numpy.asarray(dataTrain)
    yTrain = numpy.asarray(yTrain).reshape([-1,1])
 
    beta = numpy.ones((numpy.shape(dataTrain)[1])).reshape([-1,1])
    beta = numpy.asarray(beta)
    beta0 = numpy.ones((numpy.shape(dataTrain)[1])).reshape([-1,1])
    beta0 = numpy.asarray(beta0)

    logit = sigmoide(numpy.dot(dataTrain,beta))
    funcionJ = logit - yTrain
    xT = numpy.transpose(dataTrain)

    beta = beta0 - numpy.multiply(float(alfa),(xT.dot(funcionJ)))

    j = 1
    while  convergencia(beta,beta0 ,dataTrain, j) > float(threshold):
        beta0 = beta
        logit = sigmoide(numpy.dot(dataTrain,beta0))
        funcionJ = logit - yTrain
        xT = numpy.transpose(dataTrain)
        beta = beta0 - numpy.multiply(float(alfa),(xT.dot(funcionJ)))
        j +=1

    return beta, j, dataTest, yTest, dataTrain,yTrain


def convergencia( beta, beta0, dataTrain, j):
    convergenciaAux = numpy.asarray(numpy.ones((numpy.shape(dataTrain)[1])).reshape([-1,1]))
    convergencia = 0
    i = 0

    while i < len(beta):
        convergenciaAux[i] = beta[i] - beta0[i]
        convergenciaAux[i] = convergenciaAux[i] **2 
        convergencia += convergenciaAux[i]
        i += 1  

    convergencia = math.sqrt(convergencia)

    return convergencia

def probilidad(dataTest, beta):
    dataTest = numpy.asarray(dataTest)
    yhat = sigmoide(numpy.dot(dataTest,beta))
    yhat = ((yhat > .5).astype(int) )
    return yhat

def skResults(xTest,yTest,xTrain,yTrain):
    logReg = LogisticRegression()
    logReg = logReg.fit(xTrain,numpy.ravel(yTrain))
    return logReg.score(xTest,yTest), logReg.coef_

def prediccion(yhat,y):
    y = numpy.asarray(y).reshape([-1,1]).astype(int)
    presicion = numpy.mean(yhat == y)
    return presicion * 100

if __name__ == "__main__":
    
    doc = input("Ingresar nombre del archivo del dataset: ")
    porcenTrain =  input("Ingresar porcentaje de elementos en el training set: ")
    porcenTest = input("Ingresar porcentaje de elementos en test set: ")
    alfa = input("ingresar alfa: ")
    threshold = input("ingresar valor de threshold: ")
    semilla = input("ingrear semilla random: ")

    data = lectura(doc)

    if(doc == "genero.txt"):
        data = normalizarGenero(data)
    else:
        data = normalizarBanco(data)
        
    beta, iteracion,xTest,yTest,xTrain,yTrain = gradiente(data, alfa, porcenTest,porcenTrain,threshold,semilla)
    yhat = probilidad(xTest, beta)
    
    p = prediccion(yhat,yTest )
    pSk, betaSk = skResults(xTest,yTest,xTrain,yTrain)
    Xsal = numpy.asarray(xTest)
    Ysal = numpy.asarray(yTest).reshape([-1,1])

    salida = pandas.DataFrame()
    salida['x'] = Xsal.tolist()
    salida['yhat'] = yhat.tolist()
    salida['y'] = Ysal.tolist()

    print("beta: ",beta)
    print("beta del sk learn: ",betaSk)
    print("iteraciones: ", iteracion)
    print("alfa: ", alfa)
    print("threshold: ", threshold)
    print("porcentaje de error: ", 100 - p,"%")
    print("porcentaje de error del sklearn: ", 100 - (pSk*100),"%")
 
    if(doc == "genero.txt"):
        salida.to_csv(r'DatosSalidaGenero.txt')
    else:
        salida.to_csv(r'DatosSalidaBanco.txt')

    

