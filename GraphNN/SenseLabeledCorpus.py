import os
import Filesystem as F
import logging
import Utils
import lxml.etree

##### Organizing each subcorpus into Training, Validation and Test splits (80-10-10)
##### We do not rewrite the Training part of a corpus: we save separately Validation and Test,
##### and we write down the index of the document <= 80% (training stops there)

def count_elements(elements_tag, xml_fpath):
    counter = 0
    xml_docfile = open(xml_fpath, "rb")

    for event, elem in lxml.etree.iterparse(xml_docfile):
        if elem.tag == elements_tag:
            counter = counter + 1
    return counter


# Writing to files
def write_splits_subcorpus(xml_fpath, train_root, valid_root, test_root):

    logging.info("Creating ElementTree for train_root...")
    train_tree = lxml.etree.ElementTree(train_root)
    logging.info("Writing to XML file...")
    train_tree.write(os.path.join(os.path.dirname(xml_fpath), F.FOLDER_TRAIN, os.path.basename(xml_fpath)))

    logging.info("Creating ElementTree for valid_root...")
    valid_tree = lxml.etree.ElementTree(valid_root)
    logging.info("Writing to XML file...")
    valid_tree.write(os.path.join(os.path.dirname(xml_fpath),F.FOLDER_VALIDATION, os.path.basename(xml_fpath)))

    logging.info("Creating ElementTree for test_root...")
    test_tree = lxml.etree.ElementTree(test_root)
    logging.info("Writing to XML file...")
    test_tree.write(os.path.join(os.path.dirname(xml_fpath), F.FOLDER_TEST, os.path.basename(xml_fpath)))



# Splitting a subcorpus into train, validation and test, by parsing the XML tree. Based either on documents or sentences
def organize_subcorpus(xml_fpath, train_fraction):
    xml_docfile = open(xml_fpath, "rb")

    num_docs = count_elements("document", xml_fpath)
    logging.info("The corpus at " + str(xml_fpath) + " has " + str(num_docs) + " documents")
    if num_docs <= 1:
        total = count_elements("sentence", xml_fpath)
        superelements_tag = "sentence"
    else:
        superelements_tag = "document"
        total = num_docs

    num_for_training = int(train_fraction * total)
    num_for_validation = (total - num_for_training) // 2

    logging.info("Training dataset will contain: " + str(num_for_training) + " " + superelements_tag + "s , " +
                 "Validation dataset will contain: " + str(num_for_validation) + " " + superelements_tag + "s , " +
                 "Test dataset will contain: " + str(
        total - num_for_training - num_for_validation) + " " + superelements_tag + "s")

    superelements_counter = 0
    train_root = lxml.etree.Element("root")
    valid_root = lxml.etree.Element("root")
    test_root = lxml.etree.Element("root")
    for event, elem in lxml.etree.iterparse(xml_docfile):
        if elem.tag == superelements_tag:
            superelements_counter = superelements_counter + 1
            if superelements_counter < num_for_training:
                train_root.append(elem)
                if superelements_counter % 100 == 0:
                    logging.info("Appending " + str(superelements_counter) + "-th element to train_root...")
            else:
                if superelements_counter < (num_for_training + num_for_validation):
                    valid_root.append(elem)
                    if superelements_counter % 100 == 0:
                        logging.info("Appending " + str(superelements_counter) + "-th element to valid_root...")
                else:
                    test_root.append(elem)
                    if superelements_counter % 100 == 0:
                        logging.info("Appending " + str(superelements_counter) + "-th element to test_root...")
    return train_root, valid_root, test_root



##########

def organize_splits():
    Utils.init_logging('temp.log')
    xml_fnames = ['semcor.xml', 'masc.xml', 'omsti.xml', 'raganato_ALL.xml', 'wngt.xml']
    xml_fpaths = list(map(
        lambda fname: os.path.join(F.FOLDER_TEXT_CORPUSES, F.FOLDER_UFSAC, fname), xml_fnames))
    split_directories_paths = list(map(lambda dirname: os.path.join(F.FOLDER_TEXT_CORPUSES, F.FOLDER_UFSAC, dirname),
                                       [F.FOLDER_TRAIN, F.FOLDER_VALIDATION, F.FOLDER_TEST]))
    for dirpath in split_directories_paths:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    for xml_fpath in xml_fpaths:
        train_root, valid_root, test_root = organize_subcorpus(xml_fpath, 0.8)
        write_splits_subcorpus(xml_fpath, train_root, valid_root, test_root)

##########


# read an XML in UFSAC sense-labeled format. __next__() returns the dictionary of attributes of a <word> element
def dataset_generator(xml_fpath):
    xml_docfile = open(xml_fpath, "rb")

    for event, elem in lxml.etree.iterparse(xml_docfile):
        if elem.tag == "sentence":
            yield({'surface_form':Utils.EOS_TOKEN}) # following the format
        if elem.tag == "word":
            yield(elem.attrib)
