from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import os
import sys
from src.logger import logging
from src.exception import CustomException


#from sklearn.preprocessing import StandardScaler

from src.pipeline import predict_pipeline
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


app =  Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:

        data  =  CustomData(
        Hours_Studied = float(request.form.get('Hours_Studied')),              
        Attendance = float(request.form.get('Attendance')),                 
        Parental_Involvement = request.form.get('Parental_Involvement'),       
        Access_to_Resources = request.form.get('Access_to_Resources'),         
        Extracurricular_Activities = request.form.get('Extracurricular_Activities'),  
        Sleep_Hours =float (request.form.get('Sleep_Hours')),                  
        Previous_Scores = float (request.form.get('Previous_Scores')),               
        Motivation_Level = request.form.get('Motivation_Level'),               
        Internet_Access = request.form.get('Internet_Access'),                
        Tutoring_Sessions = float(request.form.get('Tutoring_Sessions')),             
        Family_Income = request.form.get('Family_Income'),                   
        Teacher_Quality= request.form.get('Teacher_Quality'),               
        School_Type = request.form.get('School_Type'),                   
        Peer_Influence= request.form.get('Peer_Influence') ,                
        Physical_Activity= float(request.form.get('Physical_Activity')),         
        Learning_Disabilities= request.form.get('Learning_Disabilities'),          
        Parental_Education_Level= request.form.get('Parental_Education_Level'),     
        Distance_from_Home = request.form.get('Distance_from_Home'),         
        Gender= request.form.get('Gender'),                         
        #Exam_Score= float(request.form.get('Exam_Score')) 
        )

        logging.info("Form data obtained from the user")

        pred_df = data.get_data_as_data_frame()
        logging.info("Dataframe created from the user input data")
        
        print("Received Data:", pred_df)
        print("Before Prediction")
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction", results)
        return render_template('home.html', results = results[0])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port = 8000, debug = True)
        
 

