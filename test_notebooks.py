import unittest
import os
import subprocess
import sys
import logging


LOG_FORMAT = '[%(asctime)s %(levelname)s] %(message)s'
LOGGER = logging.getLogger(__file__)


def run_ipynb(path):

    nb_dir = os.path.dirname(path)
    nb_path = os.path.basename(path)
    orig_dir = os.getcwd()
    os.chdir(nb_dir)

    if (sys.version_info >= (3, 0)):
        kernel_name = 'python3'
    else:
        kernel_name = 'python2'
    #  error_cells = []

    args = ["jupyter", "nbconvert",
            "--execute", "--inplace",
            "--debug",
            "--ExecutePreprocessor.timeout=5000",
            "--ExecutePreprocessor.kernel_name=%s" % kernel_name,
            nb_path]

    if (sys.version_info >= (3, 0)):
        try:
            subprocess.check_output(args)
        except TimeoutError:
            sys.stderr.write('%s timed out\n' % path)
            sys.stderr.flush()

    else:
        subprocess.check_output(args)

    os.chdir(orig_dir)


class TestNotebooks(unittest.TestCase):

    def test_data_visualization(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        run_ipynb(os.path.join(this_dir,
                               'data_visualization.ipynb'))

    def test_Feature_Extraction(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        run_ipynb(os.path.join(this_dir,
                               'Feature-Extraction.ipynb'))

    def test_classification(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        run_ipynb(os.path.join(this_dir,
                               'classification.ipynb'))

    def test_pre_processing(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        run_ipynb(os.path.join(this_dir,
                               'pre-processing.ipynb'))

    def test_segmentetaion(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        run_ipynb(os.path.join(this_dir,
                               'segmentetion.ipynb'))

    def test_feature_importances(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        run_ipynb(os.path.join(this_dir,
                               'feature_importances.ipynb'))

    def test_classification_DL(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        run_ipynb(os.path.join(this_dir,
                               'classification_DL.ipynb'))


if __name__ == "__main__":
    unittest.main()
