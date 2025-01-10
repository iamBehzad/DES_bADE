datasets = {
    #     Data set of DGA1033 report
#    "Adult",
#    "Banana",
   "Heart",
#    "ILPD",
#   "Vehicle",
#    "Glass",
#   "Pima",
#   "Sonar",
#    "Ecoli"
#    "Wine"
#    "Audit",
#    "Banknote",
#    "Blood",
#    "Breast",
#    "Car",
#    "Datausermodeling",
#    "Faults",
#    "German",
#    "Haberman",
#    "Ionosphere",
#    "Laryngeal1",
#    "Laryngeal3",
#    "Lithuanian",
#    "Liver",
#    "Mammographic",
#    "Monk2",
#    "Phoneme",
#    "Pima",
#    "Sonar",
#    "Statlog",
#    "Steel",
#    "Thyroid",
#    "Vertebral",
#    "Voice3",
#    "Weaning",
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