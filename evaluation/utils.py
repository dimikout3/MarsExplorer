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
