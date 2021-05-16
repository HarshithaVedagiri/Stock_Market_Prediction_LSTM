import pickle
import numpy as np

from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from keras.models import load_model


app = Flask(__name__)
model = load_model('model.h5')
modelScaler = pickle.load(open('scaler1.pkl','rb'))


@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():

    import pandas_datareader as pdr
    key="613e326124e4bc913a40eac66f133572b2f62811"  
    df = pdr.get_data_tiingo('AAPL', api_key=key)
    df.to_csv('AAPL.csv')
    import pandas as pd
    df=pd.read_csv('AAPL.csv')
    df1=df.reset_index(drop = True)['close']
    

    import numpy as np
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(df1, test_size=0.35, random_state=42)
    x_input=test_data[340:].values.reshape(1,-1)
    x_input.shape
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    lst_output=[]
    n_steps=100
    i=0
    inputFromUi = 30
    while(i<inputFromUi):
        
        if(len(temp_input)>100):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            # print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            # print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            # print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1

    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler(feature_range=(0,1))

    predictionInversed = modelScaler.inverse_transform(lst_output)

    
    return render_template('index.html', predicted_text="1st day output [{}]  and {}th day output {}".format(lst_output[0],i,yhat))


if __name__ == "__main__":
    app.run(debug=True)
