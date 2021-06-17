from skorch.helper import SliceDict
from entity_embeddings.models import entity_encoding_classification, get_category_count
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import torch
from skorch import NeuralNetBinaryClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

SEED = 789

if __name__ == "__main__":
    df = pd.read_csv('/Users/jonnylaw/git/entity_embeddings/titanic.csv')

    # Derive title column
    df['Title'] = df['Name'].str.extract('([A-Za-z]+\.)', expand = False)

    # Count the occurences of Title by category
    # Filter low occurences (less than or equal to 3?)
    cat_counts = get_category_count(df, ["Title"])
    rare_titles = [k for k, v in cat_counts[0].items() if v < 3]
    df['Title'].replace(to_replace=rare_titles, value='other', inplace=True)

    # Select the columns to include
    include = ['Sex', 'Age', 'Fare', 'Title']
    x = df[include]
    y = df['Survived']

    # Define the numeric and categorical columns
    num_cols = ['Fare', 'Age']
    cat_cols = ['Sex', 'Title']


    # Split the data into training and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=SEED)

    # Interpolate the numeric colums using KNN and scale using the StandardScaler
    num_standard = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])

    # Write a pipeline to 
    preprocessor = ColumnTransformer(
    transformers=[
        ("num_std", num_standard, num_cols),
        ("ordinal_encode", OrdinalEncoder(), cat_cols),
        ]
    )
    
    preprocessed = preprocessor.fit_transform(x_train, y=None)
    preprocessed_df = pd.DataFrame(preprocessed, columns=num_cols + cat_cols)

    # Define the Pytorch model
    model = entity_encoding_classification(x_train, cat_cols, 1)

    net = NeuralNetBinaryClassifier(
        module = model,
        iterator_train__shuffle=True
    )

    # To pass multiple arguments to the forward method of the PyTorch model
    # we must specify a SliceDict such that Skorch can access the data and pass 
    # it to the module properly
    Xs = SliceDict(
        x_cat=preprocessed_df[cat_cols].to_numpy(dtype="long"), 
        x_con=torch.tensor(preprocessed_df[num_cols].to_numpy(), dtype=torch.float)
    )

    # fit the classifier    
    net.fit(Xs, y=torch.tensor(y_train, dtype=torch.float))

    # pre-process the test data by re-using the pipelines from the training data
    x_test_pre = preprocessor.transform(x_test)
    preprocessed_test = pd.DataFrame(x_test_pre, columns=num_cols + cat_cols)

    # Test performance
    Xs_test = SliceDict(
        x_cat=preprocessed_test[cat_cols].to_numpy(dtype="long"), 
        x_con=torch.tensor(preprocessed_test[num_cols].to_numpy(), dtype=torch.float)
    )

    accuracy = net.score(Xs_test, y_test)

    print(accuracy)
