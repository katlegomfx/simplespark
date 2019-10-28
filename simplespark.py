
# coding: utf-8

# Katlego Madisha's spark simplification attempt package


def get_count(df):
    return df.count()
def get_catagorical_cols(df):
    return [t[0] for t in df.dtypes if t[1] == 'string']
def get_numeric_cols(df):
    return [item[0] for item in df.dtypes if item[1].startswith('int') | item[1].startswith('double')]
def get_time_cols(df):
    return [t[0] for t in df.dtypes if t[1] == 'timestamp']

def get_pipeline_cols(df, label, prim_key):
    cat_features = []
    num_features = []
    for x in df.columns:
        if x not in [label, prim_key]:
            if x in get_catagorical_cols(df):
                cat_features.append(x)
            elif x in get_numeric_cols(df):
                num_features.append(x)
        else:
            pass
    return cat_features, num_features

def drop_unwanted_cols(df, time = True, unwanted=[]):
    if time==True:
        for x in get_time_cols(df):
            df = df.drop(x)
    for x in unwanted:
        df =df.drop(x)
    return df

def fill_empty(df):
    df = df.na.fill(0)
    df = df.na.fill('none')
    return df

def make_pipeline(df, label, prim_key):
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, MinMaxScaler

    indexers = [ StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
                 for c in get_pipeline_cols(df, label, prim_key)[0] ]
    assembler = VectorAssembler(inputCols=[indexer.getOutputCol() for indexer in indexers]
                                + get_pipeline_cols(df, label, prim_key)[1], outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                            withStd=True, withMean=False)
    pipeline = Pipeline(stages=indexers + [assembler] + [scaler])
    model=pipeline.fit(df)
    return model.transform(df)

