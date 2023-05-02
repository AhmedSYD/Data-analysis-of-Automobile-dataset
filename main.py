import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import csv
import sys
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import chi2_contingency
import os


def chisq_of_df_cols(df, c1, c2):

    crosstab=pd.crosstab(df[c1], df[c2])
    # print("crose tab:",crosstab)
    # fillna(0) is necessary to remove any NAs which will cause exceptions
    return(chi2_contingency(crosstab))
def get_correlation_with_attr_categ(dataframe,attributName):
    df_new=dataframe.copy()
    ls_corr_categ_with_attr={}
    df_new[attributName]=pd.cut(dataframe[attributName], 5)
    # print("value counts=\n",df_new[attributName].value_counts())
    # print("df_new[price]=\n",df_new[attributName])
    ##get all category attributes:
    cols = list(dataframe.select_dtypes([object]).columns) ##name of all nominal attributes
    for colName in cols:
        print("colName=",colName)
        chi_out=chisq_of_df_cols(df_new, attributName, colName)
        ls_corr_categ_with_attr[colName]=[chi_out[0]]
    # print("ls_corr_categ_with_attr=",ls_corr_categ_with_attr)
    corr_categ_df = pd.DataFrame.from_dict(ls_corr_categ_with_attr)
    return corr_categ_df

def get_correlation_with_attribute(dataframe,attr_name,filesave):
    with pd.ExcelWriter(filesave) as writer:
        out_person_corr = dataframe.corr(method='pearson')
        index=out_person_corr.columns.get_loc(attr_name)
        print("index=",index)
        numeric_corr_with_attr = out_person_corr.iloc[index, :]
        numeric_corr_with_attr.to_excel(writer, sheet_name='numeric_correlation')
        # print(chisq_of_df_cols(df, "aspiration", "fuel-type"))
        out_categ_corr = get_correlation_with_attr_categ(dataframe, attr_name)
        out_categ_corr.to_excel(writer, sheet_name='categorical_correlation')
        print("out_categ_corr=", out_categ_corr)

def normalized_attribute_from_csv(dataFrame,filename,file_save):
    df_stat=pd.read_csv(filename)
    df_new=dataFrame.copy()
    n=len(df_stat.index)
    print("n=",n)
    attribute_names=[]
    with pd.ExcelWriter(file_save) as writer:
        for i in range(0,n):
            [attr_name,min,max,mean,rg,std]=list(df_stat.loc[i,:])
            print("min=",min)
            print("range=", rg)
            df_new[attr_name]=(dataFrame[attr_name]-min)/(rg)
            print("min_max_mean:",attr_name)
            print(df_new[attr_name])
            attribute_names.append(attr_name)
        # print("attr_names=", df_stat)
        df_new.to_excel(writer,columns=attribute_names, sheet_name='min_max_norm')


        for i in range(0,n):
            [attr_name,_,_,mean,_,std]=list(df_stat.loc[i,:])
            df_new[attr_name]=(dataFrame[attr_name]-mean)/(std)
            print("zscore_mean:",attr_name)
            print(df_new[attr_name])
        df_new.to_excel(writer,columns=attribute_names, sheet_name='z_score_norm')
    # print(df_stat)

def get_and_save_satistics(dataframe, attributeNames,filename):
    statistic_file = open(filename, 'w+', newline='')

    with statistic_file:
        write = csv.writer(statistic_file)
        write.writerow(["","min","max","mean","range","standard_deviation"])

        for attributeName in attributeNames:
            mean=dataframe[attributeName].mean(skipna=True)
            min = dataframe[attributeName].min(skipna=True)
            max = dataframe[attributeName].max(skipna=True)
            std = dataframe[attributeName].std(skipna=True)
            range=max-min
            print("min=", min)
            print("max=", max)
            print("mean=", mean)
            print("range=", range)
            print("std=", std)
            write.writerow([attributeName, min, max, mean, range,std])


def get_nan_indexes(dataframe, attr_name):
    return dataframe[dataframe[attr_name].isnull()].index.tolist()

def mean_by_group(dataframe, attr_name, indexes_list, number_of_vals=10,fill_dataFrame=False):
    n=len(list(dataframe[attr_name]))
    means=[]
    for index in indexes_list:
        if((index-(number_of_vals/2))<0):
            min_index =0
        else:
            min_index=index-(number_of_vals/2)
        print("min_index=",min_index)
        if((index+(number_of_vals/2))>(n-1)):
            max_index = n-1
        else:
            max_index = index +(number_of_vals / 2)

        print("max_index=", max_index)
        print("group of values=",dataframe.loc[min_index:max_index,attr_name].dropna())
        val=round(dataframe.loc[min_index:max_index,attr_name].dropna().mean())

        if fill_dataFrame:
            dataframe.loc[index,attr_name]=val
        print("val=",val)
        means.append(val)
    if fill_dataFrame:
        return means,dataframe
    else:
        return means

def nan_analysis(dataframe, attr_name ,analysis_type="replace",value_name='', null_indexs=[]):
    df_new=dataframe.copy()
    if (analysis_type=="replace"):
        if value_name=="mean":
            val = dataframe[attr_name].dropna().mean()
            print("mean val=",val)
        elif value_name=="median":
            val = dataframe[attr_name].dropna().median()
            print("median val=", val)
        elif value_name.startswith("mean_group"):
            ##get mean and fill it inside the fuction to make the runtime more faster
            num_vals = int(value_name.split("_")[-1])
            val_list,df_new = mean_by_group(df_new, attr_name, null_indexs, number_of_vals=num_vals,fill_dataFrame = True)
            print("mean_group val=", val_list)
        else:
            val=0
            print("no specified val=", val)
        ##fill data
        if(not(value_name.startswith("mean_group"))):
            df_new[attr_name]=df_new[attr_name].fillna(val)

    if(analysis_type=="drop"):
        df_new[attr_name]=df_new[attr_name].dropna()
    return df_new

def draw_specific_attribute(dataframe,attribute_name,highlight_indexes=[],filename=None):
    price_attr = dataframe[attribute_name]

    plt.figure()
    price_attr.plot(x=list(range(1, 207)),marker='o', color='black')
    if(len(highlight_indexes)>0):
        # specific_val_dataframe=dataframe.loc[highlight_indexes,:]
        # df = pd.DataFrame({"indexes":highlight_indexes,"price":list(specific_val_dataframe['price'])})
        # df.plot.scatter(x="indexes",y="price")
        price_attr.loc[highlight_indexes].plot(x=list(range(1, 207)), marker='o', color='red',linestyle="None")
    if(not(filename is None)):
        plt.savefig(filename+ ".jpg")
    plt.show()

    plt.cla()
    plt.close()


def get_and_save_number_missing(filename,df):
    num_missing_file = open(filename, 'w+', newline='')

    with num_missing_file:
        write = csv.writer(num_missing_file)
        for col in df.columns:
            num = df[df[col] == '?'].shape[0]
            print(col, num)
            write.writerow([col, num])

def draw_box_plots(cols,df, foldername=None, filename=None, visualize=False):
    print("columns inside fn",df.columns)
    for i in range(0, len(cols), 1):
        fig = plt.figure()
        boxplt = df.boxplot(column=cols[i])
        # plt.show(boxplt)

        if(filename is None):
            plt.title(cols[i])
            if(foldername is None):
                fig.savefig(cols[i] + "_boxplot.jpg")
            else:
                fig.savefig(os.path.join(foldername,cols[i] + "_boxplot.jpg"))
        else:
            plt.title(filename)
            fig.savefig(filename+ ".jpg")


        if(visualize):
            plt.show()
        plt.cla()
        plt.close()

def draw_histogram(df,binsno=3, filename=None,visualize=False):
    fig = plt.figure()
    hist = df["horsepower"].hist(bins=binsno)
    fig.text(0.25,0.85, 'Low', fontsize=12)
    fig.text(0.5, 0.85, 'Medium', fontsize=12)
    fig.text(0.75, 0.85, 'High', fontsize=12)
    if(not(filename is None)):
        fig.savefig(filename+ ".jpg")
    if (visualize):
        plt.show()
    plt.cla()
    plt.close()

def draw_scatters(df,x_name,y_name,filename=None,visualize=False):
    fig= df.plot.scatter(x=x_name,y=y_name).get_figure()
    if(not(filename is None)):
        fig.savefig(filename+ ".jpg")
    if (visualize):
        plt.show()
    plt.cla()
    plt.close()

def create_xcel_all_statistics_of_data(df):
    with pd.ExcelWriter('task2_output.xls') as writer:
        means = df.mean(axis=0, skipna=True, numeric_only=True)
        print("mean=", means)
        means.to_excel(writer, sheet_name='means')
        medians = df.median(axis=0, skipna=True, numeric_only=True)
        print("median=", medians)
        medians.to_excel(writer, sheet_name='medians')
        modes = df.mode(axis=0, dropna=True, numeric_only=True)
        print("mode=", modes)
        modes.to_excel(writer, sheet_name='modes')
        stds = df.std(axis=0, skipna=True, numeric_only=True)
        print("std=", stds)
        stds.to_excel(writer, sheet_name='stds')
        mins = df.min(axis=0, skipna=True, numeric_only=True)
        print("mins=", mins)
        mins.to_excel(writer, sheet_name='mins')
        maxs = df.max(axis=0, skipna=True, numeric_only=True)
        print("maxs=", maxs)
        maxs.to_excel(writer, sheet_name='maxs')
        qantile25 = df.quantile(q=0.25, numeric_only=True)  ##this function already skip nan
        print("qantile25=", qantile25)
        qantile25.to_excel(writer, sheet_name='qantile25')
        qantile50 = df.quantile(q=0.5, numeric_only=True)  ##this function already skip nan
        print("qantile50=", qantile50)
        qantile50.to_excel(writer, sheet_name='qantile50')
        qantile75 = df.quantile(q=0.75, numeric_only=True)  ##this function already skip nan
        print("qantile75=", qantile75)
        qantile75.to_excel(writer, sheet_name='qantile75')

def create_folder(foldername):
    # Check if the folder exists
    if not os.path.exists(foldername):
        # if it doesn't exist, creat it
        os.makedirs(foldername)

def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    ##extract data from csv files
    df=pd.read_csv("assets/imports-85.csv")

    ## creat output folder
    create_folder("out_files")
    create_folder("out_figures")
    ##count number of missing in each attributes
    get_and_save_number_missing('out_files/number_of_missing2.csv',df)

    df=df.replace('?',np.nan)
    ##change data type of numeric attributes from "object" dtype to "float" dtype.
    df=df.astype({"normalized-losses":float, "bore":float,"stroke":float,"horsepower":float,"peak-rpm":float,"price":float})
    print(df.dtypes)  ##print data type for all columns





    null_indexes_price=get_nan_indexes(df,"price")
    # mean_by_group(df, "price", null_indexes_price, number_of_vals=10)
    # sys.exit(0)
    #task1
    instructions=["mean_group_30"]
    for instruction in instructions:
        if instruction=="drop":
            df_new=nan_analysis(df,"price",analysis_type=instruction,value_name='')
        else:

            df_new = nan_analysis(df, "price", analysis_type="replace", value_name=instruction,null_indexs=null_indexes_price)
        draw_specific_attribute(df_new,"price", highlight_indexes=null_indexes_price,filename="out_figures/nan_"+instruction+"_price")
    with pd.ExcelWriter("out_files/fill_missing_price_by_mean_group.xls") as writer:
        df_new.to_excel(writer, sheet_name='fill_missing_price_byMeanGroup')

    #################
    #task2:
    cols=list(df.select_dtypes([np.int64,np.float64]).columns)
    draw_box_plots(cols,df,foldername="out_figures")
    ###############################
    ##task 3:


    get_and_save_satistics(df,["length","compression-ratio"],"out_files/length_compressionRatio_statistic.csv")
    normalized_attribute_from_csv(df,"out_files/length_compressionRatio_statistic.csv","out_files/normalization_length_ratio.xls")
    ###############################
    ##task 4:
    # get_correlation_with_attribute(df,"price","out_files/correlation_with_price.xls")
    # with pd.ExcelWriter('task2_output.xls') as writer:
    #     out_person_corr=df.corr(method ='pearson')
    #     numeric_corr_with_price=out_person_corr.iloc[-1,:-1]
    #     numeric_corr_with_price.to_excel(writer, sheet_name='numeric correlation with price')
    #     # print(chisq_of_df_cols(df, "aspiration", "fuel-type"))
    #     out_categ_corr=get_correlation_with_attr_categ(df,"price")
    #     out_categ_corr.to_excel(writer, sheet_name='categorical correlation with price')
    #     print("out_categ_corr=",out_categ_corr)

if __name__=="__main__":
    main()






