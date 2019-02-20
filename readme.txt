  _____   _____  __          __        _        _                                                                      _____                _                
 |  __ \ / ____| \ \        / /       | |      | |                     /\                                             |  __ \              | |               
 | |  | | (___    \ \  /\  / /__  _ __| | _____| |__   ___  _ __      /  \   _ __ ___   __ _ _______  _ __    ______  | |__) |___  __ _  __| |_ __ ___   ___ 
 | |  | |\___ \    \ \/  \/ / _ \| '__| |/ / __| '_ \ / _ \| '_ \    / /\ \ | '_ ` _ \ / _` |_  / _ \| '_ \  |______| |  _  // _ \/ _` |/ _` | '_ ` _ \ / _ \
 | |__| |____) |    \  /\  / (_) | |  |   <\__ \ | | | (_) | |_) |  / ____ \| | | | | | (_| |/ / (_) | | | |          | | \ \  __/ (_| | (_| | | | | | |  __/
 |_____/|_____/      \/  \/ \___/|_|  |_|\_\___/_| |_|\___/| .__/  /_/    \_\_| |_| |_|\__,_/___\___/|_| |_|          |_|  \_\___|\__,_|\__,_|_| |_| |_|\___|
                                                           | |                                                                                               
                                                           |_|                
1. User Parameters:
    The user should only interact with the following parameters, which are tagged with the '#@param' comments:

    1.1. data_source [String] - choose data source from which data will be fetched. Can be either 'Google Drive' (when running in Google's Colab env.) or 'Local'.
                                when 'Local', the notebook will download all review database dircetly to root folder, under 'data*' sub-directory.
    1.2. load_non_neutral_datasets [Boolean] - when true, loads review database without untagged reviews. (full FB size ~80GB vs no-neutral DB size ~15GB).
    1.3. use_precalculated_sizes [Boolean] - when true, uses precalculated DB size for loading DB '.tsv' files. Otherwise, calculates sizes dynamically.
    1.4. pick_category [String] - choose review category to run the DS flow on. when 'All Categories', the notebook train and test set will be created by 
                                  sampling data from all available categories. (46 in total)
    1.5. entry_limit [Integer] - limits the number of reviews which are loaded from the initial DB.
    1.6. read_random_entries [Boolean] - when true, reads random lines from DB when loading reviews. (DB lines are sorted by date).
    1.7. train_min_date [String: yyyy-mm-dd] - the minimum date for train set reviews.
    1.8. train_max_date  [String: yyyy-mm-dd] - the maximmum date for train set reviews.
    1.9. test_min_date [String: yyyy-mm-dd] - the minimum date for test set reviews.
    1.10. override_test_set [Boolean] - when true, regenerates test set (should be set true when changing params 1.8, 1.9, 1.11, 1.12). Otherwise, same test set is used for all runs.
    1.11. test_set_size [Integer] - size of test set used for model evaluation.
    1.12. test_set_percentage [Float] - sets test set relative size out of total test and train set size. 
    1.13. correct_bias [Boolean] - when true, corrects 'helpful' bias in train set. 
    1.14. save_reviews [Boolean] - when true, saves train/test dataframe to a cached variable prior to model run.
    1.15. load_reviews [Boolean] - when true, loads train/test dataframe from a cached variable prior to model run. 
                                   can be used to skip all prior cells when testing/ debugging model
    1.16. maximum_scores_on_screen [Integer] - limits the amount of feature scores displayed on screen.

* For recreating figures, scores as seen in the attached pdf, verify load_non_neutral_datasets is set to True.