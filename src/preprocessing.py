'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

import pandas as pd

def preprocessing():
#Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
    pred_universe= pd.read_csv("data/pred_universe_raw.csv")
    arrest_events= pd.read_csv("data/arrest_events_raw.csv")

#Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
    df_arrests= pd.merge(pred_universe, arrest_events, how= "outer", on= "person_id")
    df_arrests["arrest_date_univ"] = pd.to_datetime(df_arrests["arrest_date_univ"])
    df_arrests["arrest_date_event"] = pd.to_datetime(df_arrests["arrest_date_event"])

#Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
    check_future_felony= ((df_arrests["arrest_date_event"] <= df_arrests["arrest_date_univ"] + pd.Timedelta(days=365) ) &
                       (df_arrests["charge_degree"] == "felony")&
                       (df_arrests["arrest_date_event"] > df_arrests["arrest_date_univ"]))
    df_arrests["mark_future_felony"]= check_future_felony.astype(int)
    df_arrests["y"]= (df_arrests.groupby(["arrest_id_x"])["mark_future_felony"]).transform(max)

#Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
    calc_rearrest= df_arrests.drop_duplicates("person_id")["y"].mean()
    print("What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?")
    print(f"{calc_rearrest} is the share of arrestees that were rearrested for a felony crime in the next year")

#Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
    df_arrests["current_charge_felony"]= (df_arrests["charge_degree"] == "felony").astype(int)

#Use a print statment to print this question and its answer: What share of current charges are felonies?
    calc_felonies= df_arrests["current_charge_felony"].mean()
    print("What share of current charges are felonies?")
    print(f"{calc_felonies} is the share of current charges that are felonies")

#Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
    check_past_felony= ((df_arrests["arrest_date_event"] >= df_arrests["arrest_date_univ"] - pd.Timedelta(days=365) ) &
                       (df_arrests["charge_degree"] == "felony")&
                       (df_arrests["arrest_date_event"] < df_arrests["arrest_date_univ"]))
    df_arrests["mark_past_felony"]= check_past_felony.astype(int)
    df_arrests["num_fel_arrests_last_year"]= df_arrests.groupby("arrest_id_x")["mark_past_felony"].transform("sum").astype(int)

#Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
    avg_felony= df_arrests["num_fel_arrests_last_year"].mean()
    print("What is the average number of felony arrests in the last year?")
    print(f"{avg_felony} is the average number of felony arrests in the last year")

#Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
    pred_universe["num_fel_arrests_last_year"]= df_arrests.groupby("arrest_id_x")["mark_past_felony"].transform("sum").astype(int)
    print("The mean of 'num_fel_arrests_last_year' is:")
    print(pred_universe["num_fel_arrests_last_year"].mean())

#Print pred_universe.head()
    print("pred_universe.head() is:")
    print(pred_universe.head())

#Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
    df_arrests.drop(columns=["mark_future_felony", "mark_past_felony"], inplace=True)
    df_arrests.to_csv("data/df_arrests.csv", index=False)
    
    return df_arrests


# import the necessary packages



# Your code here



