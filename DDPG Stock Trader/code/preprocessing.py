import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This data preprocessing file must produce relevant momentum-like price indicators because otherwise the model might fail to cature price trends
# This step is akin to feature engineering

def plot(data):
    print(data.summary())