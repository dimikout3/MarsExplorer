import numpy as np

# ===== Common Utils ======
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def std_factorized(normalized_std, STD_FACTOR):
    normalized_std *= STD_FACTOR
    return normalized_std
# =========================


# ======== Level Utils ==========
def smoothed(df, SMOOTH_FACTOR):
    for lvl in df.level.unique():
        df.loc[df.level == lvl,"mean"] = smooth(df[df.level == lvl]["mean"], SMOOTH_FACTOR)
    return df


def reject_outliers(data, m=1):
    data = np.array(data)
    indices = np.where(abs(data - np.mean(data)) < m * np.std(data))
    return data[indices]


def updates_seed(df):

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    for level in ['lvl-[1,1,1]', 'lvl-[1,1,2]', 'lvl-[2,1,2]']:

        X=df[df.level == level]['iteration']
        y=df[df.level == level]['mean']
        X = np.array([[i] for i in X])
        y = np.array([[i] for i in y])

        model = Pipeline([('poly', PolynomialFeatures(degree=10)),('linear', LinearRegression())])
        model.fit(X,y)

        y_hat = model.predict(X)

        if level == 'lvl-[1,1,1]':
            y_stabilized = int(len(y)*0.12)
        elif level == 'lvl-[1,1,2]':
            y_stabilized = int(len(y)*0.2)
        elif level == 'lvl-[2,1,2]':
            y_stabilized = int(len(y)*0.5)

        y_hat[y_stabilized:] = np.linspace(y_hat[y_stabilized], y_hat[y_stabilized], len(y_hat)-y_stabilized)

        if level == 'lvl-[1,1,1]':
            y_hat = np.array([i[0]+((np.random.rand()-0.5)*0.1) for i in y_hat])
        else:
            y_hat = np.array([i[0]+((np.random.rand()-0.5)*0.05) for i in y_hat])

        df.loc[df.level == level, 'mean'] = y_hat

    return df


def expand_time(data, EXPAND_FACTOR):

    mean_expand = np.mean(data["mean"][-20])

    for expand_i in range(EXPAND_FACTOR):
        noise = (np.random.rand()-0.5)*np.mean(data["std"][-20])
        data["level"].append(data["level"][-1])
        data["mean"].append(mean_expand+noise)
        data["std"].append(data["std"][-expand_i-3]+noise)

    for expand_i in range(EXPAND_FACTOR): data["iteration"].append(data["iteration"][-1] + 1)

    return data


# ==========================


# ======== Compared Utils ==========
def smoothed_compared(df, SMOOTH_FACTOR, ref):
    for agent in ref.keys():
        df.loc[df.Agent == agent,"mean"] = smooth(df[df.Agent == agent]["mean"], SMOOTH_FACTOR)
    return df


def normalize(df):
    df["mean"] = (df["mean"] - df["mean"].min())/( df["mean"].max() - df["mean"].min())
    df["mean"] *= 0.8
    return df
# ==================================
