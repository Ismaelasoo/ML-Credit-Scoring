
import pandas as pd

def head_tail ( df, n ):
    separator = pd.DataFrame([["..."] * df.shape[1]], columns=df.columns)
    return pd.concat([df.head(n), separator, df.tail(n)], ignore_index = True)
