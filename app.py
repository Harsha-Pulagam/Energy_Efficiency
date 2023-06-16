from flask import Flask, render_template, request,jsonify
import os
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

app = Flask(__name__)

@app.route("/",methods=['GET','POST'])
def hello_world():
    if request.method == 'GET':

        return render_template('index.html')
    else:

        data=CustomData(
        X1 = float(request.form.get('compactness')),
        X2 = float(request.form.get('surface-area')),
        X3 = float(request.form.get('wall-area')),
        X4 = float(request.form.get('roof-area')),
        X5 = float(request.form.get('overall-height')),
        X6 = float(request.form.get('orientation')),
        X7 = float(request.form.get('glazing-area')),
        X8 = float(request.form.get('glazing-distribution')),
        
        )
        
        final_new_data=data.get_data_as_dataframe()
        print(final_new_data)
        predict_pipeline=PredictPipeline()
        heating_model_path = os.path.join('artifacts', 'model_for_heating.pkl')
        cooling_model_path = os.path.join('artifacts', 'model_for_cooling.pkl')
        heating_load =predict_pipeline.predict(features=final_new_data, model_path=heating_model_path)
        cooling_load = predict_pipeline.predict(features=final_new_data, model_path=cooling_model_path)
        # results=round(pred[0],2)

        return render_template('results.html', heating_load=heating_load, cooling_load=cooling_load)   



if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)