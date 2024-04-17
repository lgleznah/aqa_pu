import numpy as np
import pandas as pd

from functools import wraps

def negate(f):
    @wraps(f)
    def g(*args,**kwargs):
        return not f(*args,**kwargs)
    g.__name__ = f'negate({f.__name__})'
    return g

def drop_not_features(df):
    drop_columns = [col_name for col_name in df.columns if "__feature__" not in col_name]
    return df.drop(columns=drop_columns)

def pn_test_split(
        df,
        reliable_positive_fn,
        positive_fn,
        test_frac,
        random_state=1234):
    
    rng = np.random.default_rng(seed=random_state)

    df_positive = df[df.apply(lambda x: positive_fn(x, df), axis=1)]
    df_negative = df[df.apply(negate(lambda x: positive_fn(x, df)), axis=1)]

    test_positive_idxs = rng.choice(len(df_positive), size=int(test_frac * len(df_positive)), replace=False)
    test_negative_idxs = rng.choice(len(df_negative), size=int(test_frac * len(df_negative)), replace=False)

    df_test_positive = df_positive.iloc[test_positive_idxs]
    df_test_negative = df_negative.iloc[test_negative_idxs]

    # Fix to measure PU metrics properly on a true PU test set
    df_test_pu = pd.concat([df_test_positive, df_test_negative])
    y_test_pu = df_test_pu.apply(lambda x: reliable_positive_fn(x, df_test_pu), axis=1).astype(int)

    df_positive = df_positive.reset_index(drop=True)
    df_negative = df_negative.reset_index(drop=True)
    df_positive = df_positive.drop(test_positive_idxs)
    df_negative = df_negative.drop(test_negative_idxs)
    df = pd.concat([df_positive, df_negative])

    df_train_positive = df[df.apply(lambda x: reliable_positive_fn(x, df), axis=1)]
    df_train_unlabeled = df[df.apply(negate(lambda x: reliable_positive_fn(x, df)), axis=1)]

    positive_data = drop_not_features(df_train_positive).to_numpy()
    unlabeled_data = drop_not_features(df_train_unlabeled).to_numpy()

    test_positive_data = drop_not_features(df_test_positive).to_numpy()
    test_negative_data = drop_not_features(df_test_negative).to_numpy()
    X_test = np.concatenate([test_positive_data, test_negative_data])
    y_test = np.concatenate([np.ones(len(test_positive_data)), np.zeros(len(test_negative_data))])

    return positive_data, unlabeled_data, X_test, y_test, y_test_pu



def build_pu_data(
        data, 
        frac, move_to_unlabeled_frac, 
        val_split, val_split_positive,
        test_frac,
        reliable_positive_fn,
        positive_fn,
        random_state=1234):
    '''
    Divide given positive and unlabeled data into train and test splits

    Parameters
    ----------
    data: data samples
    frac: fraction of all the data to keep
    move_to_unlabeled_frac: after doing the validation split, fraction of positive data to mark as unlabeled
    val_split: fraction of all data to have as the validation split
    val_split_positive: fraction of positive data to be known positives.
    test_frac: fraction of data to keep as test set (positives and negatives)
    reliable_positive_fn: bool function that determines if an example is positive (True) or unlabeled (False)
    positive_fn: bool function that determines if an example is positive (True) or negative (False)
    random_state: seed for random splitting

    "test_split_positive" can be 'same', which keeps the original proportion of positive VS unlabeled data in the 
    test split. This is especially useful if there is significant imbalance between positive and unlabeled examples.

    Returns
    -------
    The dataset of positive and unlabeled examples
    '''
    positive_data, unlabeled_data, X_test, y_test, y_test_pu = pn_test_split(data, reliable_positive_fn, positive_fn, test_frac, random_state)
    rng = np.random.default_rng(seed=random_state)

    # Compute image amounts on each partition
    total_amount = int((len(positive_data) + len(unlabeled_data)) * frac)
    val_amount = int(total_amount * val_split)

    if (val_split_positive == 'same'):
        val_split_positive = len(positive_data) / (len(positive_data) + len(unlabeled_data))
    
    else:
        if ((am := int(val_amount * val_split_positive) > len(positive_data)) or 
            (val_amount - am) > len(unlabeled_data)):
            raise ValueError('A validation split positive fraction was specified, but there are not enough positive or unlabeled examples')

    val_positive_amount = int(val_amount * val_split_positive)
    val_unlabeled_amount = val_amount - val_positive_amount

    remaining_positive_amount = len(positive_data) - val_positive_amount
    remaining_unlabeled_amount = len(unlabeled_data) - val_unlabeled_amount
    known_positive_amount = int(remaining_positive_amount * (1.0 - move_to_unlabeled_frac))
    positive_in_unlabeled_amount = int(remaining_positive_amount * move_to_unlabeled_frac)
    unlabeled_amount = remaining_unlabeled_amount + positive_in_unlabeled_amount

    print(f'Total amount: {total_amount}')
    print(f'Validation amount: {val_amount}')
    print(f'Val_positive amount: {val_positive_amount}')
    print(f'Val_unlabeled amount: {val_unlabeled_amount}')
    print(f'Train amount: {remaining_positive_amount + remaining_unlabeled_amount}')
    print(f'Known_positive amount: {known_positive_amount}')
    print(f'Unlabeled_amount: {unlabeled_amount}')

    print(f'Size of positive paths: {len(positive_data)}')
    print(f'Size of unlabeled paths: {len(unlabeled_data)}')

    # Build test split from positive and unlabeled images
    X_val_idxs = rng.choice(len(positive_data), size=val_positive_amount, replace=False)
    X_val_positive = positive_data[X_val_idxs]
    remaining_positive = np.delete(positive_data, X_val_idxs, axis=0)

    X_val_u_idxs = rng.choice(len(unlabeled_data), size=val_unlabeled_amount, replace=False)
    X_val_unlabeled = unlabeled_data[X_val_u_idxs]
    remaining_unlabeled = np.delete(unlabeled_data, X_val_u_idxs, axis=0)

    # Build training split from both positive and unlabeled images
    X_train_positive_move_idxs = rng.choice(len(remaining_positive), size=positive_in_unlabeled_amount, replace=False)
    X_train_positive_move = remaining_positive[X_train_positive_move_idxs]
    remaining_positive = np.delete(remaining_positive, X_train_positive_move_idxs, axis=0)

    X_train = np.concatenate([remaining_positive, remaining_unlabeled, X_train_positive_move])
    y_train = np.concatenate([np.ones(len(remaining_positive)), np.zeros(len(remaining_unlabeled) + len(X_train_positive_move))])

    X_val = np.concatenate([X_val_positive, X_val_unlabeled])
    y_val = np.concatenate([np.ones(len(X_val_positive)), np.zeros(len(X_val_unlabeled))])

    #assert len(X_train) == known_positive_amount + positive_in_unlabeled_amount + true_unlabeled_amount, \
    #        f"Partition lengths wrong! Train split should have {known_positive_amount + positive_in_unlabeled_amount + true_unlabeled_amount} rows, found {len(X_train)}"

    return X_train, X_val, X_test, y_train, y_val, y_test, y_test_pu