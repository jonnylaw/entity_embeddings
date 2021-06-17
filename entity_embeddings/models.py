import torch.nn as nn
import torch
from collections import Counter
from skorch.helper import SliceDict
import numpy as np
from sklearn.preprocessing import LabelEncoder

class EmbeddingClassification(nn.Module):
    """Embed a single categorical predictor
    
    Keyword Arguments:
    
    num_output_classes: int
    num_cat_classes: list[int]
    num_cont: int
    embedding_dim: int
    hidden_dim: int
    """
    def __init__(self, num_output_classes, num_cat_classes, num_cont, embedding_dim=64, hidden_dim=64):
        super().__init__()
        # Create an embedding for each categorical input
        self.embeddings = nn.ModuleList([nn.Embedding(nc, embedding_dim) for nc in num_cat_classes])
        self.fc1 = nn.Linear(in_features=len(num_cat_classes) * embedding_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=num_cont, out_features=hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(2 * hidden_dim, num_output_classes)
        
    def forward(self, x_cat, x_con):
        # Embed each of the categorical variables
        x_embed = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_embed = torch.cat(x_embed, dim=1)
        x_embed = self.fc1(x_embed)
        x_con = self.fc2(x_con)
        x = torch.cat([x_con, x_embed.squeeze()], dim=1)
        x = self.relu(x)
        return self.out(x)    


def preprocess_training(x, y, categorical_columns, preprocess_pipeline=None):
    """pre-processes the data for the entity encoding model
    
    Use label encoder to encode the columns
    Put the data into a SliceDict since the forward method of EmbeddingClassification 
    expects two arguments (x_con, x_cat).
    Preprocesses the continuous columns using preprocess_cont_pipeline
    
    Keyword Arguments:
    x
    y
    categorical_columns
    preprocess_cont_pipeline
    """
    if preprocess_pipeline:
        x = preprocess_pipeline.fit_transform(x, y)
        x_cat = x[categorical_columns]
        x_con = x.drop(categorical_columns, axis=1)
    else:
        x_cat = x[categorical_columns]
        x_con = x.drop(categorical_columns, axis=1)

    return SliceDict(x_cat=x_cat, x_con=torch.tensor(x_con, dtype=torch.float))


def get_category_count(x, categorical_columns):
    return [Counter(x[c]) for c in categorical_columns]


def get_categorical_dimensions(x, categorical_columns):
    count_of_classes = get_category_count(x, categorical_columns)
    return [len(count) for count in count_of_classes]


def entity_encoding_classification(x, categorical_columns, num_classes):
    """
    Convenience function for the EmbeddingClassification model which     
    
    Keyword Arguments:
    x: pandas df
    y: target column
    categorical_columns: list[int] a list of the indices of the categorical columns
    num_classes: int the number of output classes of the target column
    """
    x_con = x.drop(categorical_columns, axis=1)
    categorical_dimension = get_categorical_dimensions(x, categorical_columns)

    return EmbeddingClassification(num_classes, categorical_dimension, len(x_con.columns))