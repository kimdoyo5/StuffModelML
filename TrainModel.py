import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import optuna
import seaborn as sns


def preprocess_data(df, target_column, encoder=None, feature_list=None):
    """
    Process the data so that it can be accepted by XGBoost.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[target_column])

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if feature_list is not None:
        X = X[feature_list]

    cat_cols = X.select_dtypes(include=['object', 'category']).columns

    if encoder is None:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        encoder.fit(X[cat_cols])

    X[cat_cols] = encoder.transform(X[cat_cols])

    return X, y, encoder


def calculate_accuracy(y_true, y_pred, threshold=0.2):
    """
    Calculate the percentage of predictions within a given threshold of the true value.
    """
    within_threshold = np.abs(y_true - y_pred) <= threshold * np.abs(y_true)
    accuracy = np.mean(within_threshold) * 100
    return accuracy


def plot_performance(epoch_performance, threshold, trial_number):
    """
    Plot training and validation accuracy and loss over epochs using seaborn.
    """

    epochs = list(range(1, len(epoch_performance['train']['accuracy']) + 1))
    fig = plt.figure(figsize=(16, 5))

    ax1 = plt.subplot(1, 2, 1)
    sns.lineplot(x=epochs, y=epoch_performance['train']['accuracy'], label='Training Accuracy')
    sns.lineplot(x=epochs, y=epoch_performance['val']['accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel(f'Accuracy (%) (within ±{int(threshold * 100)}%)')
    plt.title(f'Trial {trial_number}: Training vs Validation Accuracy Over Epochs')

    max_epoch = len(epochs)
    tick_positions = [1, max_epoch // 4, max_epoch // 2, (3 * max_epoch) // 4, max_epoch]
    plt.xticks(tick_positions)

    ax2 = plt.subplot(1, 2, 2)
    sns.lineplot(x=epochs, y=epoch_performance['train']['rmse'], label='Training RMSE')
    sns.lineplot(x=epochs, y=epoch_performance['val']['rmse'], label='Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title(f'Trial {trial_number}: Training vs Validation RMSE Over Epochs')
    plt.xticks(tick_positions)

    plt.tight_layout()
    plt.show()


def remove_highly_correlated_features(X_train, X_val, X_test, threshold=0.95):
    """
    Remove features that are highly correlated with each other from the datasets.
    """
    corr_matrix = X_train.corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    X_train_reduced = X_train.drop(columns=to_drop)
    X_val_reduced = X_val.drop(columns=to_drop)
    X_test_reduced = X_test.drop(columns=to_drop)

    return X_train_reduced, X_val_reduced, X_test_reduced


def objective(trial, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Objective function for Optuna hyperparameter optimization.
    """
    param = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': 42,
        'verbosity': 0,
        'tree_method': 'gpu_hist',
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    evals_result = {}

    model = xgb.train(
        param,
        dtrain,
        num_boost_round=10000,
        evals=[(dtrain, 'train'), (dval, 'eval')],
        early_stopping_rounds=50,
        evals_result=evals_result,
        verbose_eval=False
    )

    y_val_pred = model.predict(dval, iteration_range=(0, model.best_iteration + 1))

    rmse = np.sqrt(np.mean((y_val - y_val_pred) ** 2))

    num_epochs = len(evals_result['train']['rmse'])
    train_accuracies = []
    val_accuracies = []
    train_rmses = []
    val_rmses = []

    accuracy_threshold = 0.2

    for epoch in range(num_epochs):
        y_train_pred = model.predict(dtrain, iteration_range=(0, epoch + 1))
        y_val_pred_epoch = model.predict(dval, iteration_range=(0, epoch + 1))

        train_acc = calculate_accuracy(y_train, y_train_pred, threshold=accuracy_threshold)
        val_acc = calculate_accuracy(y_val, y_val_pred_epoch, threshold=accuracy_threshold)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        train_rmse = evals_result['train']['rmse'][epoch]
        val_rmse = evals_result['eval']['rmse'][epoch]

        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)

    y_test_pred = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
    test_accuracy = calculate_accuracy(y_test, y_test_pred, threshold=accuracy_threshold)

    trial.set_user_attr('train_accuracies', train_accuracies)
    trial.set_user_attr('val_accuracies', val_accuracies)
    trial.set_user_attr('train_rmses', train_rmses)
    trial.set_user_attr('val_rmses', val_rmses)
    trial.set_user_attr('test_accuracy', test_accuracy)

    return rmse


def trial_callback(study, trial):
    """
    Callback function to plot accuracies and losses after each trial.
    """
    train_accuracies = trial.user_attrs.get('train_accuracies')
    val_accuracies = trial.user_attrs.get('val_accuracies')
    train_rmses = trial.user_attrs.get('train_rmses')
    val_rmses = trial.user_attrs.get('val_rmses')
    test_accuracy = trial.user_attrs.get('test_accuracy')

    if train_accuracies is not None and val_accuracies is not None:
        epoch_performance = {
            'train': {'accuracy': train_accuracies, 'rmse': train_rmses},
            'val': {'accuracy': val_accuracies, 'rmse': val_rmses}
        }
        accuracy_threshold = 0.2

        plot_performance(epoch_performance, accuracy_threshold, trial.number)

        final_train_accuracy = train_accuracies[-1]
        final_val_accuracy = val_accuracies[-1]
        final_train_rmse = train_rmses[-1]
        final_val_rmse = val_rmses[-1]

        print(f"Trial {trial.number}:")
        print(f"  Train Accuracy: {final_train_accuracy:.2f}% (within ±{int(accuracy_threshold * 100)}%)")
        print(f"  Validation Accuracy: {final_val_accuracy:.2f}% (within ±{int(accuracy_threshold * 100)}%)")
        print(f"  Test Set Accuracy: {test_accuracy:.2f}% (within ±{int(accuracy_threshold * 100)}%)")
        print(f"  Train RMSE: {final_train_rmse:.4f}")
        print(f"  Validation RMSE: {final_val_rmse:.4f}\n")


def main():
    # train_file = 'Dataset/data/train_set.csv'
    # val_file = 'Dataset/data/val_set.csv'
    # test_file = 'Dataset/data/test_set.csv'
    #
    # train_data = pd.read_csv(train_file)
    # val_data = pd.read_csv(val_file)
    # test_data = pd.read_csv(test_file)
    #
    # data = pd.concat([train_data, val_data, test_data], ignore_index=True)
    #
    # combined_data_file = 'Dataset/data/combined_data.csv'
    # data.to_csv(combined_data_file, index=False)
    # print(f"Combined data saved to {combined_data_file}")

    data = pd.read_csv('Dataset/data/combined_data.csv')

    target_column = 'RA9'

    feature_list = [
        'stand',  # if the batter is left/right handed (categorical)
        'strikes',  # how many strikes the hitter has (categorical)
        'release_speed',  # how fast the pitch is
        'release_extension',  # how far the release point is from the rubber
        'release_pos_z',  # how high the release point is
        'release_pos_x',  # how far horizontally the release point is from the center
        'plate_z',  # vertical location of the pitch (where it ends)
        'plate_x',  # horizontal location of the pitch (where it ends)
        'az',  # vertical acceleration of the pitch
        'ax',  # horizontal acceleration of the pitch
        'estimated_arm_angle',
        'estimated_gyro_deg',
        'p_throws'
    ]

    encoder = None
    X, y, encoder = preprocess_data(data, target_column, encoder, feature_list)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1111, random_state=42)

    X_train, X_val, X_test = remove_highly_correlated_features(X_train, X_val, X_test, threshold=0.90)

    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, X_test, y_test),
        n_trials=50,
        callbacks=[trial_callback]
    )

    print("Best hyperparameters: ", study.best_params)

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': 42,
        'verbosity': 0,
        'tree_method': 'gpu_hist',
        **study.best_params
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    evals_result = {}
    epoch_performance = {
        'train': {'accuracy': [], 'rmse': []},
        'val': {'accuracy': [], 'rmse': []}
    }

    accuracy_threshold = 0.2

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=10000,
        evals=[(dtrain, 'train'), (dval, 'eval')],
        evals_result=evals_result,
        early_stopping_rounds=50,
        verbose_eval=False
    )

    for epoch in range(len(evals_result['train']['rmse'])):
        y_train_pred = model.predict(dtrain, iteration_range=(0, epoch + 1))
        y_val_pred = model.predict(dval, iteration_range=(0, epoch + 1))

        train_acc = calculate_accuracy(y_train, y_train_pred, threshold=accuracy_threshold)
        val_acc = calculate_accuracy(y_val, y_val_pred, threshold=accuracy_threshold)

        epoch_performance['train']['accuracy'].append(train_acc)
        epoch_performance['val']['accuracy'].append(val_acc)

        train_rmse = evals_result['train']['rmse'][epoch]
        val_rmse = evals_result['eval']['rmse'][epoch]

        epoch_performance['train']['rmse'].append(train_rmse)
        epoch_performance['val']['rmse'].append(val_rmse)

    plot_performance(epoch_performance, accuracy_threshold, 'Final Model')

    y_test_pred = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
    test_accuracy = calculate_accuracy(y_test, y_test_pred, threshold=accuracy_threshold)

    print("\nTest Results:")
    print(f"Accuracy: {test_accuracy:.2f}% (within ±{int(accuracy_threshold * 100)}%)")


if __name__ == '__main__':
    main()
