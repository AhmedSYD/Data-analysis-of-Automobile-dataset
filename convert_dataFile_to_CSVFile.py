import pandas as pd
def convert_data_file_to_csv_file(data_path):
    read_file = pd.read_csv (data_path,header = None)
    read_file.columns = ['symboling', 'normalized-losses', "make","fuel-type",\
                         "aspiration", "num-of-doors","body-style","drive-wheels","engine-location",\
                         "wheel-base","length","width","height","curb-weight",\
                         "engine-type","num-of-cylinders","engine-size","fuel-system","bore",\
                         "stroke","compression-ratio","horsepower","peak-rpm","city-mpg",\
                         "highway-mpg","price"]
    read_file.to_csv(data_path[:data_path.rfind(".")]+".csv", index=None)


if __name__=="__main__":
    convert_data_file_to_csv_file("assets/imports-85.data")