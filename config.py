datasets = {
    # 'Adult',
    # 'Audit', 
    # 'Banana',
    # 'Blood', 
    # 'Breast', 
    # 'Car', 
    # 'Cardiotocography',
    # 'Chess', 
    # 'Credit-screening',
    # 'CTG', 
    # 'Datausermodeling', 
    # 'Ecoli', 
    # 'Faults', 
    # 'German',
    # 'Glass', 
    # 'Haberman', 
    # 'Heart', 
    # 'ILPD',
    # 'lonosphere', 
    # 'Liver',
    # 'Magic', 
    # 'Phoneme',
    # 'Pima', 
    # 'Sonar', 
     'Thyroid', 
    # 'Transfusion',
    # 'Vehicle', 
    # 'Weaning', 
    # 'Wine' 
    }

datasets = sorted(datasets)

ExperimentPath = "./Experiment/"
NO_classifiers =100
no_itr = 20
generate_pools = True
do_train = True
do_evaluate = True

methods_names = ['KNORA-U', 'KNORAE', 'DESKNN', 'OLA', 'LCA', 'MLA', 'MCB', 'Rank', 'KNOP', 'META-DES', 'SingleBest', 'Oracle']
NO_techniques = len(methods_names)
OTHERS = True