import torch
import pandas as pd
def get_scale(ratio_start,ratio_end):
    def get_df(df):
        n = df.shape[0]
        n_start = int(n*ratio_start)
        n_end = int(n*ratio_end)+1
        return df.iloc[n_start:n_end].copy()
    return get_df

def get_vi_args(df,g):
    res_dict = {}
    res_dict["G"] = g
    res_dict["U"] = df.u.unique().size
    res_dict["L"] = df.l.unique().size
    res_dict["W"] = len(set(",".join(df.tag).split(",")))
    res_dict["T"] = df.t.unique().size
    res_dict["R"] = df.shape[0]
    res_dict["lenW"] = len(df.tag[0].split(","))
    
    return res_dict

def get_data(df):
    data = df.iloc[:,1:].to_dict("list")
    for i in ["u","t","l"]:
        data[i] = torch.tensor(data[i])

    tag_list = data["tag"]
    tag_list = [i.split(",") for i in tag_list]
    tag_list = [[int(i) for i in each] for each in tag_list]
    data["tag"] = torch.tensor(tag_list).T
    return data
    


class IdsData:
    
    def __init__(self,file_name,g):
        self.g = g
        self.ids_df = pd.read_csv("./filtered_ids/attribute-Buda.filtered.txt",names = ["pid","u","t","l","tag"])
    
    def divide_dataset(self, ratio):
        self.ratio = ratio
        self.df = self.ids_df.groupby(
            "u").apply(get_scale(0.0,self.ratio)).reset_index(drop=True)
    
    def get_training_set(self):
        return get_data(self.df),get_vi_args(self.df,self.g)
