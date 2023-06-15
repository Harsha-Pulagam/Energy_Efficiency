from flask import Flask, render_template, request,jsonify

app = Flask(__name__)

@app.route("/",methods=['GET','POST'])
def hello_world():
    if request.method == 'GET':

        return render_template('index.html')
    else:
        '''
        data=CustomData(
        Compactness=float(request.form.get('compactness')),
        Surface_area = float(request.form.get('surface-area')),
        Wall_area= float(request.form.get('wall-area')),
        Roof_area = float(request.form.get('roof-area')),
        Overall_height = float(request.form.get('overall-height')),
        Orientation = float(request.form.get('orientation')),
        Glazing_area = float(request.form.get('glazing-area:')),
        Glazing_distribution= float(request.form.get('glazing_distribution')),
        
        )
        
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)'''

        return render_template('results.html')#final_result=data)   



if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)