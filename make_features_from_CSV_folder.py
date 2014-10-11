"""
Given a folder of CSV files, extracts the data from all wikis described by each file, and parses them into feature files.

Usage: 
  python make_features_from_CSV_folder.py --outfolder feats_folder_underscore_added --csv-folder CSV_datasets

"""

import os
from argparse import ArgumentParser
from multiprocessing import Pool

import extract_wiki_data as ewd
from google_doc_csv_to_id_csv import transform


LANGUAGES_TO_ITERATE_OVER = ["English"]
# EXTRA_FILE_LABEL is a string appended to the end ot the feature file made, in order to 
# distinguish it from files made on different runs with different settings.  Examples:
# "_Unicode_parser" or "_with_X_feature"
EXTRA_FILE_LABEL = ""

LANGUAGE_NAME_TO_CODE = {"German": "de",
                         "English": "en",
                         "Chinese": "zh",
                         "French": "fr",
                         "Japanese": "ja",
                         "Polish": "pl",
                         "Portuguese": "pt",
                         "Russian": "ru",
                         "Spanish": "es"
                         }


def get_args():
    ap = ArgumentParser()
    ap.add_argument(u'--outfolder', dest=u'outfolder', default=u'corresponding_feature_data')
    ap.add_argument(u'--csv-folder', dest=u'csv_folder')
    return ap.parse_args()


"""
def extract_features_from_CSV(filename):
    if filename == ".DS_Store": # macs are odd                                                                                                                                                        
        print "Encountered .DS_Store file...skipping"            
        return
    language_name = filename.split('-')[1].split('.')[0].strip()
    if language_name in ['French', 'Spanish', 'Russian']:
        return
    print u"extracting data for %s...."%language_name
    lang = LANGUAGE_NAME_TO_CODE[language_name]
    ids = transform(open('%s/%s'%(args.csv_folder, filename), 'r'), for_secondary = False).split('\n')
    ids = ids[1:]
    ids = [int(pair.split(',')[0]) for pair in ids if pair!=',']
    ewd.extract_features_from_list_of_wids(wid_list=ids, outfile = "%s/%s_feats.csv"%(args.outfolder, lang), lang = lang)
    print u"wrote feature file for %s."%language_name
    return
"""

def main():
    args = get_args()
    if not os.path.exists(args.outfolder): #note: this can fail in race conditions, but that's liable never to be a problem here....
        os.makedirs(args.outfolder)
        
    #p = Pool(processes=len(LANGUAGE_NAME_TO_CODE.keys()))
    #filenames = os.listdir(args.csv_folder)
    #p.map(extract_features_from_CSV, filenames)
           
    for filename in os.listdir(args.csv_folder): #TODO: thread
        if filename == ".DS_Store": # macs are odd
            print "Encountered .DS_Store file...skipping"
            continue
        #reinstate below lines if there's a folder of different language data!
        #language_name = filename.split('-')[1].split('.')[0].strip()
        #if language_name not in LANGUAGES_TO_ITERATE_OVER:
        #    print "Skipping %s..."%language_name
        #    continue
        #lang = LANGUAGE_NAME_TO_CODE[language_name]
        language_name = filename
        lang = 'en'
        print u"extracting data for %s...."%language_name
        ids = transform(open('%s/%s'%(args.csv_folder, filename), 'rU'), for_secondary = False).split('\n')
        ids = ids[1:]
        ids = [int(pair.split(',')[0]) for pair in ids if pair!=',']
        step = 5000
        lb = 0
        ub = lb
        while lb<len(ids):
            print "lb: %s"%lb
            lb = ub
            ub += step
            # ewd.extract_features_from_list_of_wids(wid_list=ids, outfile = "%s/%s_feats%s.csv"%(args.outfolder, lang, EXTRA_FILE_LABEL), lang = lang)
            ewd.extract_features_from_list_of_wids(wid_list=ids[lb:ub], outfolder = "%s/%s_feats_%s%s-%s"%(args.outfolder, lang, EXTRA_FILE_LABEL, lb, ub), lang = lang)
               
if __name__ == u'__main__':
    main()

