Here are the scripts that are provided in this folder:

1. predict_classes.py - the main character in this ensemble.  Given a csv of wids
to predict labels for, outputs a csv with the predicted classes for new wiki, as
well as the predicted probability for each vertical, and the prediction cetainty.
If the user specifies, relearns the models based on the training set given 
(takes a couple hours as I recall).  In this case you'll want to have run make_features_from_CSV_folder.py
Otherwise, uses the classifiers saved in trained_things.  These were experimentally 
determined to be the best using a script not contained in this folder.

2. make_features_from_CSV_folder.py - use this script if you are training the 
algorithms on new data.

3. csv_aggregator is actually useful.  Use this in the case that you run 
predict_classes.py on some absurd amount of wikis, and decide to break the 
output up into lots of little csvs, that you then want to combine

4. trained_things - contains pretrained models and vectorizers.

5. extract_wiki_data.py - used by make_features_from_CSV_folder.py; you probably
don't have to worry about this.

6. google_doc_csv_to_id_csv.py - is called by predict_classes.py and 
make_features_from_CSV_folder.py; you don't have to worry about it.


I have not included the scripts for comparing different ensembles for accuracy on
 a held out set; I can give them to whomever asks.

