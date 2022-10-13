
## Inspiration
Hospital Readmissions are a critical problem to address for hospitals.  Readmissions affect hospitals both Clinically and Financially. The hospitals are penalized due to 30-day readmission cases. 

As per CMS guidelines, [link](https://www.cms.gov/Medicare/Medicare-Fee-for-Service-Payment/AcuteInpatientPPS/Readmissions-Reduction-Program) 
 - CMS began penalizing hospitals for 30-day readmissions on Oct. 1, 2012, at 1 percent, upping the penalty rate to 2 percent for the fiscal year 2014
 - CMS will cut payments to the penalized hospitals by as much as 3 percent for each Medicare case during fiscal 2020, which runs from Oct. 1 through September 2020

CMS includes the following condition or procedure-specific 30-day risk-standardized unplanned readmission measures in the program:
 - Acute myocardial infarction (AMI)
 - Chronic obstructive pulmonary disease (COPD)
 -  Heart failure (HF)
 -  Pneumonia
 -  Coronary artery bypass graft (CABG) surgery
 -  Elective primary total hip arthroplasty and/or total knee arthroplasty (THA/TKA)

If we can build a predictive model to predict the re-admission cases in advance, hospitals can take preventive action and take special care of those patients with higher re-admission risks.
Also Model should be able to predict the top influencing factors which increase the re-admission risks. Hospitals can focus on these critical factors and plan to prevent the re-admission.


## What it does

### Project Objective:

 - Build a model which can predict 30-Day Re-admission cases for Heart Failure ICU Patients.
 - The model will identify the top factors which increase the re-admission risks in heart failure ICU patients.


## How we built it

### Data Strategy

**MIMIC III Dataset** : MIMIC-III is a large, freely-available database comprising de-identified health-related data associated with over forty thousand patients who stayed in critical care units of the Beth Israel Deaconess Medical Center between 2001 and 2012. 

The database includes information such as :

 - patient demographics and in-hospital mortality.
 - laboratory test results (for example, hematology, chemistry, and microbiology results).
 - discharge summaries and reports of electrocardiogram and imaging studies.
 - billing-related information such as the International Classification of Disease, 9th Edition (ICD-9) codes, Diagnosis Related   Group (DRG) codes, and 
 - Current Procedural Terminology (CPT) codes.

**MIMIC-III tables used**:
 lab events, lab items, patients, admissions, noteevents
 
 
#### Methods

**Extract Tabular data**: Get all labitems for all the patients. 
 - We used Heart failure ICD-9 codes to extract the heart failure patients' details.
 - The codes used ->
39891, 40201, 40211, 40291, 40401, 40403, 40411, 40413, 40491, 40493, 4280, 4281, 42820,42821, 42822, 42823, 42830, 42831, 42832, 42833, 42840, 42841 
 - The readmission cases are selected where the number of admissions is more than one with first admission and 30-days duration. 
 - The labitems have multiple reports for each patient and admission entry.
 - Hence we have grouped multiple reports with average report values. 

**Extract unstructured data:**
 - The discharge summaries are extracted from NOTEEVENTS. 
 - Then the extract is merged with tabular patient-admission reports and attached to heart failure ICU patients.
 - The discharge summary text is extracted and processed by AWS Comprehend Medical.
 - We used detect entity detect_entities_v2() function to process the text.
 - boto3 library is used to connect to the AWS account.

**AWS Comprehend Medical**

Here is the main function used to connect AWS Comprehend Medical and extract the entity information of DX_NAME where we have selected the top 10 text which is diagnosis names.


Code:
`

def comprih_extract(input_txt):

    import boto3

    import pandas as pd

    print('doc size', len(input_txt))
   
    input_txt = input_txt[0:20000]
    client = boto3.client(service_name='comprehendmedical',region_name='us-east-1')
    result = client.detect_entities_v2(Text=input_txt)
    tmp_df = pd.DataFrame(result['Entities'])

    return_val = list((tmp_df[tmp_df.Type=='DX_NAME']['Text'].value_counts()[0:10]).reset_index()['index'].values)
    
    if len(return_val) == 0:
        return_val += ['nan'] * (N-len(return_val)+1)
    else:
        return_val += ['nan'] * (N-len(return_val))

    print('length:',len(return_val))
    
    return return_val
`

Once the diagnosis names are returned in a list format for each admission, we saved the output.

**Clustering of diagnosis information using NLP Topic Modelling Technique**

 - Once the diagnosis names are returned in a list format for each admission, we saved the output.

 - In the next module, the diagnosis information is stored in a data frame and clustered using NLP Topic Modelling Technique. The AWS Comprehend  Medical required the diagnosis in a tokenized format. 

**We have applied the below NLP tasks ** 
 - unigram, bigram, and trigram analysis is done
 - converted the raw texts to a matrix of TF-IDF features
 - Create a document term matrix using fit_transform
 - The applied gensim NMF modeling.
 - Non-Negative Matrix Factorization (NMF) is an unsupervised technique so there is no labeling of topics that the model will be trained on. The way it works is that NMF decomposes (or factorizes) high-dimensional vectors into a lower-dimensional representation. These lower-dimensional vectors are non-negative which also means their coefficients are non-negative.
 - Used Gensim's NMF to get the best num of topics via coherence score
 - With the CoherenceModel we got the best number of topics=5.
 - The only parameter that is required is the number of components i.e. the number of topics we want. This is the most crucial step in the whole topic modeling process and will greatly affect how good the final topics are.
 - Then Created the best topic for each complaint
 - topic_results = nmf_model.transform(dtm)
 - Once the topics are attached dataset is saved.

#### Building Final Model with both Structured and Unstrcutured data

 - In the next module, the dataset with topics is treated as clustered diagnosis text. This is now merged with the previous labitems tabular dataset. 
 - Used this merged data and train the final model.
 
#### Model Evaluation

 - As the merged dataset is labeled with Re-admission Yes (=1) and No(=0), this is a supervised classification problem.
 - Then we applied different models like
 - DecisionTreeClassifier, LogisticRegression,  RandomForestClassifier,     MLPClassifier,   XGBClassifier.
 - The model accuracy is an average of 57% for MLPClassifier which is showing the best results. 
 - However, the accuracy is highly dependent on how much information is extracted from the unstructured data.

**Below Features are important( from higher to lower) to predict the Re-admission cases for Heart Failure ICU Patients:**

age : 0.20
Urea Nitrogen : 0.18
duration : 0.11
NTproBNP : 0.11
Sodium : 0.09
Topic : 0.05
admission_type_EMERGENCY : 0.03
religion_CATHOLIC : 0.02
insurance_Medicare : 0.02
admission_type_ELECTIVE : 0.02
gender_M : 0.02
marital_status_MARRIED : 0.01
ethnicity_WHITE : 0.01
gender_F : 0.01
marital_status_WIDOWED : 0.01
insurance_Private : 0.01
marital_status_SINGLE : 0.01
ethnicity_UNKNOWN/NOT SPECIFIED : 0.01
religion_PROTESTANT QUAKER : 0.01
admission_type_URGENT : 0.01
insurance_Medicaid : 0.01
##################################################

time_elapsed 0.17884421348571777
##################################################
MLPClassifier
-------------

Result on test set
              precision    recall  f1-score   support

           0       0.57      0.27      0.37       296
           1       0.57      0.82      0.67       342

    accuracy                           0.57       638
   macro avg       0.57      0.55      0.52       638
weighted avg       0.57      0.57      0.53       638

--------------------------------------------------
##################################################



 - Because of the sudden spike in AWS Comprehend Medical charges (~4000$) we were running out of credits (still having a discussion with AWS Support for adjustment) and could not use Comprehend for ICD-10 and other extraction processes.



## Challenges we ran into

 - As per our plan we wanted to extract other text information from discharge summaries like procedure codes and ICD-9 codes (which are having equivalent ICD-10 codes in AWS Comprehend Medical).
 - But the diagnosis text exhausted all the credits and charged a huge amount of **~4000 dollars** to our AWS Account. The same is now being investigated by the AWS Support team.We used MIMIC III database and extracted 6000 discharge summaries which were fed to Comprehend Medical for entity extraction. We ran the extraction function via boto3 and left it at night as it was taking time to run. In the morning  everything was completed.
  -When we saw the first budget alert the charge was already reached ~4K dollars.
 - We immediately checked the account and saw this huge cost.
 - We had to stop all the AWS workloads and we could not continue the text analysis on other attributes.


## Accomplishments that we're proud of

 - We could build a well-suited strategy to extract both structured and unstructured data from a clinical database and train a model.
 - The model can predict the re-admission cases in advance and help hospitals to focus on high risk patients.
 - The model will help hospitals utilize resources properly, focus on high risks patients and increase operational efficiency
 - Improve hospital rating based on lower readmission rates 
 - Increased patient satisfaction by providing an advanced patient care service
 - A positive financial return by reducing the penalties caused due to re-admission.


## What we learned

 - How to extract unstructured data from a clinical database and process quickly with AWS Comprehend Medical.
 - We have done text processing using various NLP techniques, however, we faced several challenges with  clinical data:
 - Medical jargon, Non-standard sentences, flexible formatting, unusual grammar, free text format, etc.
AWS Comprehend Medical can extract entities, ICD-10, SNOMED, and RXNORM formats within a minute and also provide an API to extract the response in JSON format.
 - The API provides various functions as per different use cases.
 - Once we extract the clinical entities, diagnosis, procedures, treatments, drugs, organ sites, drugs frequencies, and negations statements , rest of the analysis becomes generic NLP analysis.


## What's next for Predicting 30-Day Readmission from Discharge Summary

The challenges in healthcare data are
* patient protection
* data quality
* cost(monetary, time,resources)
* Transparency
* disparate rules across stakeholders

**The model accuracy is highly dependent on correct feature selection. 
A huge amount of information can be extracted from discharge summaries, clinical notes, lab reports, and Radiology reports.
As a next level, we will try to analyze ICD codes, procedure codes, drug information, patients' history, negation statements, and bias information from the discharge summaries & other ICU reports.** 
