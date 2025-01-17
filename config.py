datasets = {
        # 'Adult',
        # 'Audit',
        # 'Audit2',
        # 'Banana',
        # 'Banknote',
        # 'Blood',
         'Breast',
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
        # 'Ionosphere',
        # 'Iris',
        # 'Laryngeal1',
        # 'Laryngeal3',
        # 'Lithuanian',
        # 'Liver',
        # 'Magic',
        # 'Mammographic',
        # 'Monk2',
        # 'Phoneme',
        # 'Pima',
        # 'Segmentation',
        # 'Sonar',
        # 'Statlog',
        # 'Steel',
        # 'Thyroid',
        # 'Transfusion',
        # 'Vehicle',
        # 'Vertebral',
        # 'Voice3',
        # 'Voice9',
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