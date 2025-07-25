'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import matplotlib.pyplot as plt
import etl
import preprocessing
import logistic_regression
import decision_tree



# Call functions / instanciate objects from the .py files
def main():

    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    etl.etl()
    # PART 2: Call functions/instanciate objects from preprocessing
    df_arrests= preprocessing.preprocessing()
    # PART 3: Call functions/instanciate objects from logistic_regression
    logistic_regression.logistic_regression(df_arrests)
    # PART 4: Call functions/instanciate objects from decision_tree

    df_arrests_train= pd.read_csv("data/df_arrests_train.csv")
    df_arrests_test= pd.read_csv("data/df_arrests_test.csv")
    decision_tree.decision_tree(df_arrests_train, df_arrests_test)
    # PART 5: Call functions/instanciate objects from calibration_plot
    from calibration_plot import calibration_plot 
    plt.figure()
    calibration_plot(df_arrests_test["y"],df_arrests_test["pred_lr"] , n_bins=5)
    calibration_plot(df_arrests_test["y"],df_arrests_test["pred_dt"] , n_bins=5)
    print("Which model is more calibrated?")
    print("The decision tree model is more calibrated")

if __name__ == "__main__":
    main()