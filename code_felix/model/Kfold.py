def learning(model ,Xtrain ,y ,Xtest, number_of_folds= 5, seed = 777, nb_class =22):

    Xtrain = Xtrain.reset_index(drop=True)
    Xtest  = Xtest.reset_index(drop=True)

    print( 'Model: %s' % model)

    """ Each model iteration """
    train_predict_y = np.zeros((len(y), nb_class))
    test_predict_y = np.zeros((Xtest.shape[0], nb_class))
    ll = 0.
    """ Important to set seed """
    skf = StratifiedKFold(n_splits = number_of_folds ,shuffle=True, random_state=seed)
    """ Each fold cross validation """

    for i, (train_idx, val_idx) in enumerate(skf.split(Xtrain, y)):
        print('Fold ', i + 1)

        model.fit(Xtrain.values[train_idx], y[train_idx], eval_set=[(Xtrain.values[val_idx],  y[val_idx])],
                  early_stopping_rounds=50, verbose=True)

        scoring = model.predict_proba(Xtrain.values[val_idx])
        """ Out of fold prediction """
        train_predict_y[val_idx] = scoring
        l_score = log_loss(y[val_idx], scoring)
        ll += l_score
        print('    Fold %d score: %f' % (i + 1, l_score))

        test_predict_y = test_predict_y + model.predict_proba(Xtest)

    test_predict_y = test_predict_y / number_of_folds

    print('average val log_loss: %f' % (ll / number_of_folds))
    """ Fit Whole Data and predict """
    print('training whole data for test prediction...')

    np.save('./output/xgb_train.np', train_predict_y)
    np.save('./output/xgb_test.np', test_predict_y)



if __name__ == '__main__':

    from code_felix.model.xgb import *

    feature_label = get_stable_feature('1011')

    #feature_label = get_cut_feature(feature_label, False)

    # feature_label = get_best_feautre(feature_label)


    # daily_info = summary_daily_usage()
    # feature_label  = feature_label.merge(daily_info, on='device', how='left')

    train = feature_label[feature_label['sex'].notnull()]
    train  = reorder_train(train)

    test = feature_label[feature_label['sex'].isnull()]
    test = reorder_test(test)


    Y = train['sex_age']
    Y_CAT = pd.Categorical(Y)

    train = train.drop(['sex', 'age', 'sex_age', 'device'], axis=1, errors='ignore' )
    test = test.drop(['sex', 'age', 'sex_age', 'device'], axis=1, errors='ignore' )


    learning(get_model(), train, Y_CAT.codes, test )
