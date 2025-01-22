datasets = {
        # 'Adult',
        # 'Audit',
        # 'Banknote',
        # 'Blood',
        # 'Breast',
        # 'Car',
        # 'Cardiotocography',
        # 'Credit-screening',
        # 'Faults',
        # 'German',
        # 'Glass',
        # 'Haberman',
        # 'Heart',
        # 'ILPD',
        # 'Ionosphere',
        # 'Iris',
        # 'Laryngeal3',
        # 'Lithuanian',
        # 'Liver',
        # 'Magic',
        # 'Mammographic',
        # 'Monk2',
        # 'Pima',
        # 'Sonar',
        # 'Statlog',
        # 'Steel',
        #'Thyroid',
        # 'Transfusion',
        # 'Vehicle',
        # 'Vertebral',
        # 'Voice3',
        # 'WDVG1',
        # 'Weaning',
        # 'Wholesale',
        # 'Wine',
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