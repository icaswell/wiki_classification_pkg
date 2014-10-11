
import os

csv_folder  = "predictions_Aug_8/"

final_file = csv_folder + "agregate.csv"

with open(final_file, 'a+') as ff:
    for filename in os.listdir(csv_folder):
        print filename
        if filename == ".DS_Store" or csv_folder + filename == final_file: # macs are odd
            print "Encountered .DS_Store file...skipping"
            continue
        #with open(csv_folder + filename + '/unprocessed.csv', 'rU') as subf:
        with open(csv_folder + filename, 'rU' ) as subf:
            for line in subf:
                ff.write(line)
            
        

