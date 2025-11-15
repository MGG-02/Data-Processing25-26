folds= ['charliehebdo-all-rnr-threads','ottawashooting-all-rnr-threads',
'ebola-essien-all-rnr-threads','prince-toronto-all-rnr-threads',
'ferguson-all-rnr-threads',	'putinmissing-all-rnr-threads',
'germanwings-crash-all-rnr-threads',
'gurlitt-all-rnr-threads','sydneysiege-all-rnr-threads']


# the annotations on PHEME have 2 boolean values, 'misinformation' value and 'true' value, we want to convert them in a single label with values
# 'true', 'false' or 'unverified'. This values can be string or numerical, depending on the 'string' parameter of the function
#obtained from: https://github.com/ManuelCecere/fake_news_detection

def convert_annotations_data(annotation, string = True):
    if 'misinformation' in annotation.keys() and 'true'in annotation.keys():
        if int(annotation['misinformation'])==0 and int(annotation['true'])==0:
            if string:
                label = "unverified"
            else:
                label = 2
        elif int(annotation['misinformation'])==0 and int(annotation['true'])==1 :
            if string:
                label = "true"
            else:
                label = 1
        elif int(annotation['misinformation'])==1 and int(annotation['true'])==0 :
            if string:
                label = "false"
            else:
                label = 0
        elif int(annotation['misinformation'])==1 and int(annotation['true'])==1:
            print ("OMG! They both are 1!")
            print(annotation['misinformation'])
            print(annotation['true'])
            label = None
            
    elif 'misinformation' in annotation.keys() and 'true' not in annotation.keys():
        # all instances have misinfo label but don't have true label
        if int(annotation['misinformation'])==0:
            if string:
                label = "unverified"
            else:
                label = 2
        elif int(annotation['misinformation'])==1:
            if string:
                label = "false"
            else:
                label = 0
                
    elif 'true' in annotation.keys() and 'misinformation' not in annotation.keys():
        print ('Has true not misinformation')
        label = None
    else:
        print('No annotations')
        label = None
           
    return label