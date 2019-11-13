import os 
os.chdir("C:/Users/ruslan.bergenov/RD Local/Udacity-Machine-Learning-Introduction-Nanodegree/Unsupervised Learning/Project Customer Segmentation/Data")

# import libraries here; add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.ticker as mtick # to format x axis labels with thousand comma separators


# Load in the general demographics data.
azdias = pd.read_csv("Udacity_AZDIAS_Subset.csv", sep=';')

# Load in the feature summary file.
feat_info = pd.read_csv("AZDIAS_Feature_Summary.csv", sep=';')


# how many missing (NAN) values there are in each column
# this is BEFORE we convert categories which are not misssing values, but denote missing or unknown information
# This is how much data is NATURALLY missing (Python NAN, SQL null, Excel blank)
def how_many_NA(df):
    """Loop thru columns. Give number of missing NA values.
    
    his function tells you how many missing values there are in each column.

    if you add it up to azdias.info() non-null, you will get 891221 in each row.
    
    This is an extended version of the function above. It also calculates % of missing columns. 
    Returns a dataframe/report on NAN
    """
    missing_NA_list = []
    missing_NA_percent_list = []
    
    for column in df.columns.tolist():
        missing_NA = df[column].isna().sum()
        
        missing_NA_percent = missing_NA / len(df)
        
        missing_NA_list.append(missing_NA)
        missing_NA_percent_list.append(missing_NA_percent)
        
    missing_value_report_df = pd.DataFrame(
            {'Column': df.columns.tolist(),
             'missing_NA': missing_NA_list,
             'missing_NA_percent': missing_NA_percent_list
             }
            )
    return missing_value_report_df






def how_many_coded_uknown_or_missing(df):
    
    """"loop thru columns of df.
        Look at which specific values are codes for missing and uknown.
        How many values are coded as missing or unknown? Print report.
    """
    
    # initialize empty lists
    coded_unknown_or_missing_list = []
    coded_unknown_or_missing_percent_list = []
    i = 0

    # loop thu each column. Which values indicate missing and uknown?
    # how many values are coded as missing and uknown? (They would need to be replaced with NAN)
    for column in df.columns.tolist():
        
        # this is just the number of column
        print("Colum number:", i)
        i=i+1

        list_NAN_for_this_column = feat_info.missing_or_unknown[feat_info[feat_info.attribute == column].index[0]]

        print(column)
        
        print("missing' or 'unknown' code BEFORE parsing/cleaning the string")
        print(list_NAN_for_this_column)
        print(type(list_NAN_for_this_column))

        list_NAN_for_this_column = list_NAN_for_this_column.strip('[').strip(']').split(",")

        print("missing' or 'unknown' code AFTER parsing/cleaning the string")
        print(list_NAN_for_this_column)
        print(type(list_NAN_for_this_column))

        print("Sum of values in the original dataset which are coded as unknown or missing.")

        coded_unknown_or_missing = df[column].isin(list_NAN_for_this_column).sum()

        print(coded_unknown_or_missing)
        print("#############################################################################")
        print()
        
        
        # append sum of values are coded as missing and uknown to the list
        coded_unknown_or_missing_list.append(coded_unknown_or_missing)

        coded_unknown_or_missing_percent = coded_unknown_or_missing / len(df)

        coded_unknown_or_missing_percent_list.append(coded_unknown_or_missing_percent)    

    # create df
    
    coded_missing_unknown_report_df = pd.DataFrame(
                {
                 'Column': df.columns.tolist(),
                 'Coded_Uknown_or_Missing': coded_unknown_or_missing_list,
                 'Coded_Uknown_or_Missing_Percent': coded_unknown_or_missing_percent_list
                 }
                )
    
    return coded_missing_unknown_report_df  




def replace_coded_as_missing_unknown_with_NANs(df):
    
    """
    copy dataframe 
    
    in the copy, replace missing and uknown codes with NAN
    
    int64 gets replaced with float64, not sure why, probably cus NAN corresponds to a float
    
    https://stackoverflow.com/questions/53819909/pandas-replace-values-in-dataframe-conditionally-based-on-string-compare
    
    https://stackoverflow.com/questions/41870093/pandas-shift-converts-my-column-from-integer-to-float
    
    
    """
    
    # create a copy
    df_cleaned = df.copy()
    
    for column in df_cleaned.columns.tolist():
        
        # which values in this columnn are indicating missing or uknown?
        list_NAN_for_this_column = feat_info.missing_or_unknown[feat_info[feat_info.attribute == column].index[0]]
       
        # clean up the missing and unkown codes
        list_NAN_for_this_column = list_NAN_for_this_column.strip('[').strip(']').split(",")
                
        # replace with NAN. 2 methods. Link in docustring from Stack Overflow. Both work well and been tested
        
#        df_cleaned[column] = np.where(df_cleaned[column].isin(list_NAN_for_this_column), np.nan, df_cleaned[column])
        
        df_cleaned.loc[df_cleaned[column].isin(list_NAN_for_this_column), column] = np.nan
        
    return df_cleaned


def count_NAN_in_each_ROW(df):
    """
    count number of missing values in each row
    https://datascience.stackexchange.com/questions/12645/how-to-count-the-number-of-missing-values-in-each-row-in-pandas-dataframe"""
    # number of columns MINUS smth like Excel COUNTA function (should count both text and numbers)
    df['NAN_count'] = df.shape[1] - df.apply(lambda x: x.count(), axis=1)
    
    
    
    













# Write code to divide the data into two subsets based on the number of missing
# values in each row.

def divide_into_subsets(df_cleaned, threshold):
    
    """divide data into 2 subsets depending on how many missing values there in row
    provide threshold
    """
    condition = df_cleaned['NAN_count'] > threshold

    df_cleaned['many_missing_values_in_row'] = np.where(condition==True, 1, 0)

    print("threshold: ", threshold)
    print()

    print("percent of rows with lots of missing values: ", df_cleaned['many_missing_values_in_row'].mean())
    print()
    print(df_cleaned['many_missing_values_in_row'].describe())
    print()

    df_cleaned_missing_few = df_cleaned[df_cleaned['many_missing_values_in_row'] ==0]
    df_cleaned_missing_many = df_cleaned[df_cleaned['many_missing_values_in_row'] ==1]

    print("Few Missing Values")
    print(df_cleaned_missing_few['NAN_count'].describe())
    print()
    print("Lots of Missing Values")
    print(df_cleaned_missing_many['NAN_count'].describe())
    
    return df_cleaned_missing_few, df_cleaned_missing_many, threshold





multilevel = ['CJT_GESAMTTYP', 'FINANZTYP', 'GFK_URLAUBERTYP', 'LP_FAMILIE_FEIN', 'LP_FAMILIE_GROB', 'LP_STATUS_FEIN', 'LP_STATUS_GROB', 'NATIONALITAET_KZ', 'SHOPPER_TYP', 'ZABEOTYP', 'GEBAEUDETYP', 'CAMEO_DEUG_2015', 'CAMEO_DEU_2015']





def recode_column_according_2_my_dict(df, oldvar, newvar, my_dict):
    """recode a column according to how it's specified in my dictionary my_dict"""

    for item in my_dict:
        condition = df[oldvar].isin(my_dict[item])
        df[newvar] = np.where(condition==True, item, df[newvar])

    """
    https://stackoverflow.com/questions/33271098/python-get-a-frequency-count-based-on-two-columns-variables-in-pandas-datafra
    """
    return pd.crosstab(df[oldvar], df[newvar])






def clean_data(df):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data

    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame

    NOTE: This function reuses the data transformation functions from previous sections. This will help avoid code repetition and make this function more readable.
    """

    ###########################################################################
    # Put in code here to execute all main cleaning steps:
    # convert missing value codes into NaNs, ...
    
#    df = azdias.copy() # test - remove later. Keep commented out in PROD
    
    
    df_cleaned = replace_coded_as_missing_unknown_with_NANs(df) 
    
    # summarize NAN for columns
    df_cleaned_NA_report = how_many_NA(df_cleaned)

    # remove selected columns and rows, ...

    # Investigate patterns in the amount of missing data in each column.
    #  the following columns have more than 1/3 of data missing
    outliers_df = df_cleaned_NA_report[df_cleaned_NA_report['missing_NA_percent']>0.333]

    # delete outlier columns with too much missing data
    for outlier_column in outliers_df['Column'].tolist():
        del(df_cleaned[outlier_column])

    ###########################################################################
    # add a column to the df_cleaned with NAN count for each row - takes 3-5 mins to run
    count_NAN_in_each_ROW(df_cleaned)

    # divide into subsets
    df_cleaned_missing_few, df_cleaned_missing_many, threshold = divide_into_subsets(df_cleaned, 42)

    # select, re-encode, and engineer column values.
    #########################################################
    # Make dummy variables for OST_WEST_KZ

    df_cleaned_encoded = pd.concat([df_cleaned_missing_few, pd.get_dummies(df_cleaned_missing_few['OST_WEST_KZ'], prefix='OST_WEST_KZ')], axis=1)



    #########################################################
    # recode PRAEGENDE_JUGENDJAHRE
    # initialize column
    df_cleaned_encoded['PRAEGENDE_JUGENDJAHRE_DECADE'] = 0

    my_dict = {
    40:[1,2],
    50:[3,4],
    60:[5,6,7],
    70:[8,9],
    80:[10,11,12,13],
    90:[14,15]
    }
    
    print(my_dict)
    
    recode_column_according_2_my_dict(df_cleaned_encoded, "PRAEGENDE_JUGENDJAHRE", "PRAEGENDE_JUGENDJAHRE_DECADE",my_dict)
    
    print(pd.crosstab(df_cleaned_encoded["PRAEGENDE_JUGENDJAHRE"], df_cleaned_encoded['PRAEGENDE_JUGENDJAHRE_DECADE']))


    df_cleaned_encoded['PRAEGENDE_JUGENDJAHRE_MOVEMENT_TEMP'] = ""

    my_dict ={
        "AVANTGARDE": [2, 4, 6, 7, 9, 11, 13, 15],
        "MAINSTREAM": [1, 3, 5, 8, 10, 12, 14]
    }

    recode_column_according_2_my_dict(df_cleaned_encoded, "PRAEGENDE_JUGENDJAHRE", "PRAEGENDE_JUGENDJAHRE_MOVEMENT_TEMP",my_dict)

    # Make dummy variables for PRAEGENDE_JUGENDJAHRE_MOVEMENT_TEMP
    df_cleaned_encoded = pd.concat([df_cleaned_encoded, pd.get_dummies(df_cleaned_encoded['PRAEGENDE_JUGENDJAHRE_MOVEMENT_TEMP'], prefix='PRAEGENDE_JUGENDJAHRE_MOVEMENT')], axis=1)


    print(pd.crosstab(df_cleaned_encoded['PRAEGENDE_JUGENDJAHRE_MOVEMENT_TEMP'], df_cleaned_encoded['PRAEGENDE_JUGENDJAHRE_MOVEMENT_AVANTGARDE']))

    print(pd.crosstab(df_cleaned_encoded['PRAEGENDE_JUGENDJAHRE_MOVEMENT_TEMP'], df_cleaned_encoded['PRAEGENDE_JUGENDJAHRE_MOVEMENT_MAINSTREAM']))


    # PRAEGENDE_JUGENDJAHRE_MOVEMENT_ is caused by missing values, we don't need this col, it just tell u number of NAN in the original column PRAEGENDE_JUGENDJAHRE
    # PRAEGENDE_JUGENDJAHRE_MOVEMENT_TEMP is a temporary column, we don't need it either


    #########################################################
    df_cleaned_encoded["CAMEO_INTL_2015_WEALTH"] =df_cleaned_encoded["CAMEO_INTL_2015"].str.slice(0,1)

    df_cleaned_encoded["CAMEO_INTL_2015_LIFE_STAGE_TYP"] =df_cleaned_encoded["CAMEO_INTL_2015"].str.slice(1,2)



    #########################################################

    df_cleaned_encoded['PLZ8_BAUMAX_BLDNG_TYPE_TEMP'] = ""

    my_dict ={
    "FAMILY": [1,2,3,4],
    "BUSINESS": [5]
    }

    recode_column_according_2_my_dict(df_cleaned_encoded, "PLZ8_BAUMAX", "PLZ8_BAUMAX_BLDNG_TYPE_TEMP", my_dict)

    df_cleaned_encoded = pd.concat([df_cleaned_encoded, pd.get_dummies(df_cleaned_encoded['PLZ8_BAUMAX_BLDNG_TYPE_TEMP'], prefix='PLZ8_BAUMAX_BLDNG_TYPE')], axis=1)



    # how many family homes are there?
    condition = df_cleaned_encoded['PLZ8_BAUMAX'] == 5

    # df_cleaned_encoded['PLZ8_BAUMAX_FAMILY_HOMES'] = df_cleaned_encoded['PLZ8_BAUMAX']

    df_cleaned_encoded['PLZ8_BAUMAX_FAMILY_HOMES']  = np.where(condition==True, np.nan, df_cleaned_encoded['PLZ8_BAUMAX'])


    #########################################################

    condition = df_cleaned_encoded["WOHNLAGE"].isin([7,8])

    df_cleaned_encoded["WOHNLAGE_RURAL_FLAG"] = np.where(condition==True, 1,0)

    # condition = df_cleaned_encoded["WOHNLAGE"].isin([7,8])

    df_cleaned_encoded["WOHNLAGE_CITY_NEIGHBOURHOOD"] = np.where(condition==True, np.nan,df_cleaned_encoded["WOHNLAGE"])

    #########################################################
    # the variable multilevel comes from code above which analyzes cat variables
    df_cleaned_encoded = df_cleaned_encoded.drop(multilevel, axis = 1)

    drop_list = ['OST_WEST_KZ','PRAEGENDE_JUGENDJAHRE','PRAEGENDE_JUGENDJAHRE_MOVEMENT_TEMP','PRAEGENDE_JUGENDJAHRE_MOVEMENT_','CAMEO_INTL_2015','PLZ8_BAUMAX_BLDNG_TYPE_TEMP','PLZ8_BAUMAX_BLDNG_TYPE_','PLZ8_BAUMAX','WOHNLAGE','LP_LEBENSPHASE_FEIN','LP_LEBENSPHASE_GROB','NAN_count','many_missing_values_in_row']

    df_cleaned_encoded = df_cleaned_encoded.drop(drop_list, axis = 1)


    # Return the cleaned dataframe.
    return df_cleaned_encoded

azdias_cleaned_encoded_TEST_BIGASS_CLEANING_FUNCTION = clean_data(azdias)