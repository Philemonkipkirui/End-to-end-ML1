import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import load_objects
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path =  os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model = load_objects(file_path=model_path)
            preprocessor = load_objects(file_path=preprocessor_path)

            # Check expected vs. received columns
            if hasattr(preprocessor, "feature_names_in_"):
                expected_features = list(preprocessor.feature_names_in_)
            else:
                # Fallback if preprocessor lacks feature_names_in_ (rare case)
                num_features = preprocessor.transformers_[0][2]
                cat_features = preprocessor.transformers_[1][2]
                expected_features = num_features + cat_features
                
            received_features = list(features.columns)

            print("✅ Model expects columns:", expected_features)
            print("✅ Received columns:", received_features)

            missing_columns = set(expected_features) - set(received_features)
            if missing_columns:
                print(f"⚠️ Warning: Missing columns detected! {missing_columns}")
                for col in missing_columns:
                    features[col] = 0

            # Ensure correct column order before transformation
            features = features[expected_features]

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
        Hours_Studied : int,                 
        Attendance: int,                 
        Parental_Involvement:str,         
        Access_to_Resources: str,          
        Extracurricular_Activities :str,   
        Sleep_Hours :int,                  
        Previous_Scores:int,               
        Motivation_Level:str,               
        Internet_Access:str,                
        Tutoring_Sessions:int,            
        Family_Income:str,                  
        Teacher_Quality:str,               
        School_Type :str,                   
        Peer_Influence:str,                
        Physical_Activity:int,          
        Learning_Disabilities:str,         
        Parental_Education_Level:str,     
        Distance_from_Home :str,         
        Gender:str,                         
        #Exam_Score:int,                   
            ):
        
        self.Hours_studied = Hours_Studied
        self.Attendance = Attendance
        self.Parental_involvement = Parental_Involvement
        self.Access_to_Resources = Access_to_Resources
        self.Extracurricular_Activities = Extracurricular_Activities
        self.Sleep_Hours = Sleep_Hours
        self.Previous_Score =  Previous_Scores
        self.Motivational_level =  Motivation_Level
        self.Internet_Access =  Internet_Access
        self.Tutoring_Sessions =  Tutoring_Sessions
        self.Family_Income =  Family_Income
        self.Teacher_Quality =  Teacher_Quality
        self.School_Type =  School_Type
        self.Peer_Influence = Peer_Influence
        self.Physical_Activity =  Physical_Activity
        self.Learning_Disabilities =  Learning_Disabilities
        self.Parental_Education_Level =  Parental_Education_Level
        self.Distance_From_Home = Distance_from_Home
        self.Gender = Gender
        #self.Exam_Score = Exam_Score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict ={
                "Hours_Studied" :[self.Hours_studied],              
                "Attendance":[self.Attendance],                
                "Parental_Involvement":[self.Parental_involvement],       
                "Access_to_Resources": [self.Access_to_Resources],         
                "Extracurricular_Activities" :[ self.Extracurricular_Activities],   
                "Sleep_Hours" :[self.Sleep_Hours],                
                "Previous_Scores":[self.Previous_Score],              
                "Motivation_Level":[self.Motivational_level],               
                "Internet_Access":[self.Internet_Access ],                
                "Tutoring_Sessions":[self.Tutoring_Sessions],            
                "Family_Income":[ self.Family_Income],                 
                "Teacher_Quality":[self.Teacher_Quality] ,              
                "School_Type":[self.School_Type],                   
                "Peer_Influence":[self.Peer_Influence],                
                "Physical_Activity":[self.Physical_Activity],          
                "Learning_Disabilities":[self.Learning_Disabilities],         
                "Parental_Education_Level":[self.Parental_Education_Level],    
                "Distance_from_Home" :[self.Distance_From_Home],         
                "Gender":[self.Gender],                         
                #"Exam_Score":[self.Exam_Score] 

            }

            return pd.DataFrame (custom_data_input_dict)
        except Exception as e:
            return CustomException(e,sys)
            

        