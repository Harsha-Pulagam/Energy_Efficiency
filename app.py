from flask import Flask, render_template, request,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

app = Flask(__name__)

@app.route("/",methods=['GET','POST'])
def hello_world():
    if request.method == 'GET':

        return render_template('index.html')
    else:

        data=CustomData(
        Relative_Compactness = float(request.form.get('compactness')),
        Surface_Area = float(request.form.get('surface-area')),
        Wall_Area= float(request.form.get('wall-area')),
        Roof_Area = float(request.form.get('roof-area')),
        Overall_Height = float(request.form.get('overall-height')),
        Orientation = float(request.form.get('orientation')),
        Glazing_Area = float(request.form.get('glazing-area')),
        Glazing_Area_Distribution= float(request.form.get('glazing-distribution')),
        
        )
        
        final_new_data=data.get_data_as_dataframe()
        print(final_new_data)
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('results.html', final_result=data)   



if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)