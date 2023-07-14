import numpy as np

def build_pu_data(positive_data, unlabeled_data, frac, known_positive_frac, positive_frac_in_unlabeled, test_split, test_split_positive, random_state=1234):
    '''
    Divide given positive and unlabeled data into train and test splits

    Parameters
    ----------
    positive_data: positively labeled data samples
    unlabeled_data: data samples with unknown labels
    frac: fraction of all the data to keep
    known_positive_frac: for the training split, fraction of known positive data.
    positive_frac_in_unlabeled: for the unlabeled data in the training split, fraction of known positive data.
    test_split: fraction of all data to have as the test split
    test_split_positive: fraction of positive data to be known positives
    random_state: seed for random splitting

    Returns
    -------
    The dataset of positive and unlabeled examples
    '''
    rng = np.random.default_rng(seed=random_state)

    # Compute image amounts on each partition
    total_amount = int((len(positive_data) + len(unlabeled_data)) * frac)
    test_amount = int(total_amount * test_split)
    test_positive_amount = int(test_amount * test_split_positive)
    test_unlabeled_amount = test_amount - test_positive_amount
    train_amount = total_amount - test_amount
    known_positive_amount = int(train_amount * known_positive_frac)
    unlabeled_amount = train_amount - known_positive_amount
    positive_in_unlabeled_amount = int(unlabeled_amount * positive_frac_in_unlabeled)
    true_unlabeled_amount = unlabeled_amount - positive_in_unlabeled_amount

    # Build test split from positive and unlabeled images
    X_test_idxs = rng.choice(len(positive_data), size=test_positive_amount, replace=False)
    X_test_positive = positive_data[X_test_idxs]
    remaining_positive = np.delete(positive_data, X_test_idxs, axis=0)

    X_test_u_idxs = rng.choice(len(unlabeled_data), size=test_unlabeled_amount, replace=False)
    X_test_unlabeled = unlabeled_data[X_test_u_idxs]
    remaining_unlabeled = np.delete(unlabeled_data, X_test_u_idxs, axis=0)

    # Build training split from both positive and unlabeled images
    X_train_positive_idxs = rng.choice(len(remaining_positive), size=known_positive_amount, replace=False)
    X_train_positive = remaining_positive[X_train_positive_idxs]
    remaining_positive = np.delete(remaining_positive, X_train_positive_idxs, axis=0)

    X_train_unlabeled_positive_idxs = rng.choice(len(remaining_positive), size=positive_in_unlabeled_amount, replace=False)
    X_train_unlabeled_positive = remaining_positive[X_train_unlabeled_positive_idxs, :]
    remaining_positive = np.delete(remaining_positive, X_train_unlabeled_positive_idxs, axis=0)

    X_train_unlabeled_idxs = rng.choice(len(remaining_unlabeled), size=true_unlabeled_amount, replace=False)
    X_train_unlabeled = remaining_unlabeled[X_train_unlabeled_idxs]

    X_train = np.concatenate([X_train_positive, X_train_unlabeled_positive, X_train_unlabeled])
    y_train = np.concatenate([np.ones(len(X_train_positive)), np.zeros(len(X_train_unlabeled_positive) + len(X_train_unlabeled))])

    X_test = np.concatenate([X_test_positive, X_test_unlabeled])
    y_test = np.concatenate([np.ones(len(X_test_positive)), np.zeros(len(X_test_unlabeled))])

    assert len(X_train) == known_positive_amount + positive_in_unlabeled_amount + true_unlabeled_amount, \
            f"Partition lengths wrong! Train split should have {known_positive_amount + positive_in_unlabeled_amount + true_unlabeled_amount} rows, found {len(X_train)}"

    return X_train, X_test, y_train, y_test