import numpy as np

def build_pu_data(positive_data, unlabeled_data, frac, move_to_unlabeled_frac, test_split, test_split_positive, random_state=1234):
    '''
    Divide given positive and unlabeled data into train and test splits

    Parameters
    ----------
    positive_data: positively labeled data samples
    unlabeled_data: data samples with unknown labels
    frac: fraction of all the data to keep
    move_to_unlabeled_frac: after doing the test split, fraction of positive data to mark as unlabeled
    test_split: fraction of all data to have as the test split
    test_split_positive: fraction of positive data to be known positives.
    random_state: seed for random splitting

    "test_split_positive" can be 'same', which keeps the original proportion of positive VS unlabeled data in the 
    test split. This is especially useful if there is significant imbalance between positive and unlabeled examples.

    Returns
    -------
    The dataset of positive and unlabeled examples
    '''
    rng = np.random.default_rng(seed=random_state)

    # Compute image amounts on each partition
    total_amount = int((len(positive_data) + len(unlabeled_data)) * frac)
    test_amount = int(total_amount * test_split)

    if (test_split_positive == 'same'):
        test_split_positive = len(positive_data) / (len(positive_data) + len(unlabeled_data))
    
    else:
        if ((am := int(test_amount * test_split_positive) > len(positive_data)) or 
            (test_amount - am) > len(unlabeled_data)):
            raise ValueError('A test split positive fraction was specified, but there are not enough positive or unlabeled examples')

    test_positive_amount = int(test_amount * test_split_positive)
    test_unlabeled_amount = test_amount - test_positive_amount

    remaining_positive_amount = len(positive_data) - test_positive_amount
    remaining_unlabeled_amount = len(unlabeled_data) - test_unlabeled_amount
    known_positive_amount = int(remaining_positive_amount * (1.0 - move_to_unlabeled_frac))
    positive_in_unlabeled_amount = int(remaining_positive_amount * move_to_unlabeled_frac)
    unlabeled_amount = remaining_unlabeled_amount + positive_in_unlabeled_amount

    print(f'Total amount: {total_amount}')
    print(f'Test amount: {test_amount}')
    print(f'Test_positive amount: {test_positive_amount}')
    print(f'Test_unlabeled amount: {test_unlabeled_amount}')
    print(f'Train amount: {remaining_positive_amount + remaining_unlabeled_amount}')
    print(f'Known_positive amount: {known_positive_amount}')
    print(f'Unlabeled_amount: {unlabeled_amount}')

    print(f'Size of positive paths: {len(positive_data)}')
    print(f'Size of unlabeled paths: {len(unlabeled_data)}')

    # Build test split from positive and unlabeled images
    X_test_idxs = rng.choice(len(positive_data), size=test_positive_amount, replace=False)
    X_test_positive = positive_data[X_test_idxs]
    remaining_positive = np.delete(positive_data, X_test_idxs, axis=0)

    X_test_u_idxs = rng.choice(len(unlabeled_data), size=test_unlabeled_amount, replace=False)
    X_test_unlabeled = unlabeled_data[X_test_u_idxs]
    remaining_unlabeled = np.delete(unlabeled_data, X_test_u_idxs, axis=0)

    # Build training split from both positive and unlabeled images
    X_train_positive_move_idxs = rng.choice(len(remaining_positive), size=positive_in_unlabeled_amount, replace=False)
    X_train_positive_move = remaining_positive[X_train_positive_move_idxs]
    remaining_positive = np.delete(remaining_positive, X_train_positive_move_idxs, axis=0)

    X_train = np.concatenate([remaining_positive, remaining_unlabeled, X_train_positive_move])
    y_train = np.concatenate([np.ones(len(remaining_positive)), np.zeros(len(remaining_unlabeled) + len(X_train_positive_move))])

    X_test = np.concatenate([X_test_positive, X_test_unlabeled])
    y_test = np.concatenate([np.ones(len(X_test_positive)), np.zeros(len(X_test_unlabeled))])

    #assert len(X_train) == known_positive_amount + positive_in_unlabeled_amount + true_unlabeled_amount, \
    #        f"Partition lengths wrong! Train split should have {known_positive_amount + positive_in_unlabeled_amount + true_unlabeled_amount} rows, found {len(X_train)}"

    return X_train, X_test, y_train, y_test