from flask import Flask, render_template, request,jsonify
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
        pred=predict_pipeline.predict(features=final_new_data)

        # results=round(pred[0],2)

        return render_template('results.html', final_result=pred)   



if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)