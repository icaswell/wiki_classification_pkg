"""
@Author: Isaac
@created 4 August 2014

Trains a (n ensemble) classifier on given data, predicts labels of unlabeled data, and writes these predictions
(along with confidences) to a csv file.  Does NOT evaluate accuracy of classifiers.  can be made to function at 
an arbitrary level of granularity in terms of memory usage, as specified by the size of files inputted.  
(I.e. the input need not be one big file, which capability is necessary when we're working with e.g. 400.000 data instances)

NOTE: if you are training first, and then predicting, you need to pass in training documents as parameters.  
However, if you've already trained, merely modify the global CLASSIFIERS_IN_ENSEMBLE to contain the ids of the
classifiers in the ensemble that proved to be the best in testing.

Arguments:
  --train_csv: A csv file with wiki ids and their hand classified labels (training set). In an 
     idosyncratic file format, so check google_doc_csv_to_id_csv.transform if you're doing it yourself.
  --test-csv: A csv file (or folder of such files) of wiki ids of wikis yet to be classified
  --train-feats:  A csv file of training data (feature vectors)
  --test-feats: The same for test data
  --isfolder: a boolean value indicating whether --test-feats refers to a folder (recommended for 
     large inputs, around 20000 unknowns) or a file
  --prelearned: a boolean indicating whether the models in the ensemble have been learned already 
     from the test data.  If true, the training data is not used.
  --verbose: a value from [0,2] indicating the verbosity of the output

Result:
   writes a csv file of wiki ids with URL, predicted label, and probability assigned to all 5 labels,
   for those wikis from test-feats.  if test-feats is a folder, a separate prediction file is made for
   each file in it.
   
   All resulting files are placed in a csv file.
   (note that the input should not contain wikis with the gaming or lifestyle labels (or 'other'?), 
   as these do not change.

Example Usage:
  #First, extract features for the training set, if you're about to train the classifier: 
  [python make_features_from_CSV_folder.py --csv-folder CSV_datasets_train --outfolder training_feats]
  
  #Then get prediction on the new data:
  python predict_classes.py --train-feats training_feats.csv --train-csv CSV_datasets_train --test-feats test_feats --test-csv CSV_top_2000/en_top_2000.csv [--isfolder] [--prelearned]

More Examples:
  python predict_classes.py --train-feats training_feats_en.csv --train-csv CSV_datasets_train/en_unprocessed.csv --test-feats feats_all_wikias_segmented --test-csv CSV_all_wikias/All\ wikis.csv --prelearned --isfolder

  python predict_classes.py --train-feats training_feats_en.csv --train-csv CSV_datasets_train/en_unprocessed.csv --test-feats test_feats_top_2000.csv --test-csv CSV_top_2000/en_top_2000.csv

"""

from collections import OrderedDict, defaultdict
from argparse import ArgumentParser, FileType
import numpy as np
from time import asctime
import sys, os
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

from  __init__ import Classifiers
from google_doc_csv_to_id_csv import transform

log_level = logging.INFO
logger = logging.getLogger(u'wikia_dstk.classification')
logger.setLevel(log_level)
ch = logging.StreamHandler()
ch.setLevel(log_level)
formatter = logging.Formatter(u'%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


CLASSIFIERS_IN_ENSEMBLE = ["knn", "maxent"]
FLAG_MESSAGE = "UNCERTAIN" #The excel line is flagged with this message if the classifier is 
                           #particularly uncertain about the class of the example
TRAINED_CLF_STEM = "trained_things/trained_classifier-"

def get_args():
    ap = ArgumentParser()
    ap.add_argument('--train-feats', type=FileType('rU'), dest='train_feats')
    ap.add_argument('--train-csv', type=FileType('rU'), dest='train_csv')
    ap.add_argument('--test-feats', type=str, dest='test_feats')
    ap.add_argument('--test-csv', type=FileType('rU'), dest='test_csv')
    ap.add_argument('--isfolder', dest='isfolder', action='store_true', default = False)
    ap.add_argument('--prelearned', dest='prelearned', action='store_true', default = False)
    ap.add_argument('--verbose', dest = 'verbose', type=int, default = 1)
    return ap.parse_args()


def main():
    args = get_args()
    predict_and_write_to_file(args.train_feats, args.train_csv, args.test_feats, args.test_csv, args.isfolder, args.prelearned, args.verbose)

def prep_test_data(feat_file_test, vectorizer, verbose):
    """Transforms the feature file into a numerical representation, namely a tf-idf matrix."""
    verbose_print("loading test CSV...", verbose, 1)
    wid_to_features_test = get_feature_dict(feat_file_test)

    feature_keys_test = wid_to_features_test.keys()
    feature_rows_test = wid_to_features_test.values()
    X_test = vectorizer.transform(feature_rows_test)
    return X_test, feature_keys_test


def train_ensemble_classifiers(classifier_strings, training_vectors, vector_classes):
    """
    Trains all classifiers in classifier_strings on the training data, and saves them separately to 
    pickle files.

    :param training_vectors: a numpy array of training vectors
    :type training_vectors:class:`numpy.array`
    :param vector_classes: a list of numeric class ids for each vector, in order
    :type vector_classes: list
    :return: None
    """

    training_vectors = training_vectors.toarray()
    for classifier_string in classifier_strings:
        clf = Classifiers.get(classifier_string)
        classifier_name = Classifiers.classifier_keys_to_names[classifier_string]

        logger.info(u"Training a %s classifier on %d instances..." % (classifier_name, training_vectors.shape[0]))
        clf.fit(training_vectors, vector_classes)
        #with open(TRAINED_CLF_STEM + classifier_string + '.pkl', 'wb') as fid:
        joblib.dump(clf, TRAINED_CLF_STEM + classifier_string + '.pkl', compress=9)


def predict_ensemble_pretrained(classifiers, test_vectors):
    """
    Performs ensemble prediction for a set of already trained classifiers listed by key name in Classifiers class.
    differs from the version in __init__ in that the classifiers are already trained, to eenable more efficient decomposition.

    The nature of the ensemble prediction is simplistic: the probabilities predicted by each classifier are simply summed, and 
    the class which maximizes this sum is chosen.  The adventurous should experiment with weighting each classifier's 
    contribution by, for instance, the prediction of a metaclassifier of which classifier would work better given the data instance.
    
    :param classifiers: a non-empty list of classifier objects
    :type classifiers: list
    :param test_vectors: a numpy array of vectors to predict class for
    :type test_vectors:class:`numpy.array`
    :return: an ordered list of predicted classes for each test vector row, as well as the probabilities assigned by the ensemble to each class.
    :rtype: tuple(list, list(list))
    """
    test_vectors = test_vectors.toarray() #certain classifiers don't support sparse representation

    scores = defaultdict(lambda: defaultdict(list))
    #scores[i][clf_string][j] represents P(test instance i is of class j|clf)

    for classifier_string in CLASSIFIERS_IN_ENSEMBLE:
        clf = classifiers[classifier_string]
        logger.info(u"Predicting with %s for %d unknowns..." % (classifier_string, test_vectors.shape[0]))
        prediction_probabilities = clf.predict_proba(test_vectors)

        for i, p in enumerate(prediction_probabilities): # for each instance to be predicted
            # p[j] is the probability that instance i belongs to class j, as predicted by classifier_string
            scores[i][classifier_string].append(p) # why not "=" instaed of append?? it would make more sense.  Harmless tho

    logger.info(u"%s Predictions" % (u"Finalizing" if len(classifiers) == 1 else u"Interpolating"))
    predictions = []
    ensemble_prediction_probs = []
    for i in scores: #for each data instance to be predicted...
        combined = (np.sum(scores[i].values(), axis=0) / float(len(scores[i])))[0]
        ensemble_prediction_probs.append(combined)
        predictions.append(list(combined).index(max(combined)))
    return predictions, ensemble_prediction_probs



def predict_and_write_to_file(feat_file_train, csv_file_train, test_feats, test_csv, feats_are_in_folder, prelearned, verbose=1):
    """
    Trains an ensemble (if not prelearned), predicts the unknowns specified in test_feats,
    and writes the predictions and confidences to a file.
    The primary method exported by this script, and the functionality of main().
    :param feat_file_train: csv with lines of form wiki_id,feat1,feat2,feat3.....
    :param csv_file_train: a csv file with idiosyncratic format dealt with by google_id_csv_doc......py. Used for training class labels
    :param feat_file_test: csv with lines of form wiki_id,feat1,feat2,feat3.....
    :param test_csv:
    """
    outfolder = "predictions_%s"%date_string() #yay for explicit file labeling!
    outfile_stem =  outfolder + "/"

    if not os.path.exists(outfolder): #note: this can fail in race conditions, but that's liable never to be a problem here....
        os.makedirs(outfolder)
    else:
        print "USE A DIFFERENT OUTFOLDER!!  (you don't want to accientally end up with duplicates!)"
        exit(0)


    verbose_print("Classifying with Ensemble composed of %s...."%", ".join(CLASSIFIERS_IN_ENSEMBLE), verbose, 1)
    # note that, since we call the transform function directly, there is no need for
    # an outfile.  This saves time and space!
    # Note: below actually a list, not a file.
    class_file_train = transform(csv_file_train, for_secondary = False).split('\n')
    groups = defaultdict(list) # TODO should be called 'wids to class name' or something
    for line in class_file_train: #changed by Isaac
        splt = line.strip().split(',')
        if splt[0] != '' and splt[0].isdigit(): # added by Isaac
            groups[splt[1]].append(int(splt[0]))

    class_name_to_label = OrderedDict([(key, i) for i, key in enumerate(groups.keys())])
    
    
    verbose_print("loading train CSV...", verbose, 1)
    
    
    if not prelearned:
        #==============================#
        # PROCESS TRAIN DATA
        #==============================#
        wid_to_features_train = get_feature_dict(feat_file_train)
        #tokeep_indices = [int(f) for f in np.random.random_sample((20,))*200]#TODO REMOVE
        #tokeep = np.array(wid_to_features_train.keys())[tokeep_indices]#TODO REMOVE
        #wid_to_features_train = {k:v for k,v in wid_to_features_train.items() if k in tokeep}#TODO REMOVE
        
        assert set(wid_to_features_train.keys()) == set([str(v) for g in groups.values() for v in g]),\
            "feature file and class file refer to different wikis"
        
        verbose_print(u"Training dataset has %s data instances"%(len(wid_to_features_train.values())), verbose)
        verbose_print(u"Vectorizing...", verbose)
        
        feature_keys_train = wid_to_features_train.keys()
        feature_rows_train = wid_to_features_train.values()
        
        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(feature_rows_train)
        joblib.dump(vectorizer, TRAINED_CLF_STEM + 'vectorizer' + '.pkl')
        verbose_print("Training dataset has %s features"%vectorizer.idf_.shape, verbose)
        #NOTE: !!! for some machines (computers), like aws, one has to write "vectorizer.tfidf.idf_.shape".  No idea why
        
        wid_to_class = dict([(str(wid), class_name_to_label[key]) for key, wids in groups.items() for wid in wids])
        y_train = [wid_to_class[wid] for wid in feature_keys_train]
        
        #==============================#
        # TRAIN_CLASSIFIERS and save to file
        #==============================#
        train_ensemble_classifiers(CLASSIFIERS_IN_ENSEMBLE, X_train, y_train)
        
    #==============================#
    # LOAD TRAINED CLASSIFIERS; VECTORIZER
    #==============================#
    classifiers = {}
    for clf_str in CLASSIFIERS_IN_ENSEMBLE:
        #with open(TRAINED_CLF_STEM + clf_name + '.pkl', 'rb') as fid:
        classifiers[clf_str] = joblib.load(TRAINED_CLF_STEM + clf_str + '.pkl')

    vectorizer = joblib.load(TRAINED_CLF_STEM + 'vectorizer' + '.pkl')
    #==============================#
    # MAKE LIST OF TEST FILES TO ITERATE OVER
    #==============================#
    feat_fnames_test = []
    if not feats_are_in_folder:
        feat_fnames_test = [test_feats]
    else:
        feat_fnames_test = os.listdir(test_feats)
        feat_fnames_test = [s for s in feat_fnames_test if s[0] != '.'] #deal with .DS_store file, etc
        done = ['en_feats_245000-250000', 'en_feats_260000-265000', 'en_feats_280000-285000', 'en_feats_190000-195000',\
                    'en_feats_130000-135000', 'en_feats_40000-45000', 'en_feats_255000-260000','en_feats_115000-120000', \
                    'en_feats_155000-160000', 'en_feats_60000-65000']
        feat_fnames_test = [s for s in feat_fnames_test if s not in done]

    #==============================#
    # PREDICT AND WRITE TO FILE FOR EACH TEST FILE
    #==============================#
    other_fields = dict([(splt[1], [splt[0], splt[2]]) for splt in
                         [line.decode(u'utf8').strip().split(u',') for line in test_csv]
                         ])

    for i, feat_fname in enumerate(feat_fnames_test):
        verbose_print("opening %s..."%feat_fname, verbose, 1)
        with open("%s/%s/unprocessed.csv"%(test_feats, feat_fname), 'rU') as feat_file:
            outfile = outfile_stem + 'pt%s-%s.csv'%(i, feat_fname)
            X_test, feature_keys_test = prep_test_data(feat_file, vectorizer, verbose)
            verbose_print("predicting labels for new data...", verbose, 1)
            predictions, prediction_probs = predict_ensemble_pretrained(classifiers, X_test)
            confidences, uncertainty_flags = get_confidence_on_predictions(prediction_probs)

            verbose_print("writing csv file to %s..."%outfile, verbose, 1)
            write_prediction_csv(feature_keys_test, predictions, prediction_probs, outfile,\
                                     class_name_to_label.keys(), confidences, uncertainty_flags, other_fields)


def write_prediction_csv(feature_keys_test, predictions, prediction_probs, outfile, label_to_class_name, confidences, flags, other_fields):
    with open(outfile, "w") as f:
        #Note: the key 'Wiki ID' is an artefact of the file format, and may not work with other CSVs
        #other_fields_names_segment = ','.join(other_fields[u'Wiki ID'])
        #clf_names_segment = ','.join(label_to_class_name)
        #f.write("wid,%s,predicted class,%s,confidence,flags\n"%(other_fields_names_segment, clf_names_segment))
        for i, wid in enumerate(feature_keys_test):
            pp_csv_segment = ','.join([str(p) for p in prediction_probs[i]])
            other_fields_segment = ','.join(other_fields[str(wid)])
            line = wid + ',' + other_fields_segment + ',' + label_to_class_name[predictions[i]] + ',' + pp_csv_segment + ',' + str(confidences[i]) + ','
            line += FLAG_MESSAGE if flags[i] else ""
            line += '\n'
            f.write(line)


def get_feature_dict(feats_file):
    """returns a dict of id: features, where features is a space separated string of features."""
    wid_to_features = OrderedDict([(splt[0], u" ".join(splt[1:])) for splt in
                                   [line.decode(u'utf8').strip().split(u',') for line in feats_file]
                                   ])
    return wid_to_features


def get_confidence_on_predictions(prediction_probs, pct_to_flag=.30):
    N = len(prediction_probs)
    confidences = [certainty_margin(p) for p in prediction_probs]
    std = np.argsort(np.array(confidences))
    flags = np.zeros((N,))
    flags[std[0:N*pct_to_flag]] = 1
    return confidences, flags

def certainty_margin(arr):
    std = np.sort(arr)
    assert std[-1] - std[-2] >=0 #make sure sorting worked....
    return std[-1] - std[-2]


#========================#
#Extraneous helpers:
#========================#
def date_string():
    """returns 'Aug_5' or similar"""
    return "_".join(asctime()[4:10].split())


def verbose_print(msg, program_verbosity_level, msg_verbosity_level=1):
    """
    usage:
    verbose_print("message", verbose)
    verbose_print("really detailed message", verbose, 2)
    verbose_print("you really want me to say everything I'm doing, dontcha?", 3)
    """
    if program_verbosity_level >=msg_verbosity_level:
        print msg


#========================#
#Execute the program if called from command line!
#========================#
if __name__ == u'__main__':
    main()

