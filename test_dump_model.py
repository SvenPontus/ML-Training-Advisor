import unittest
from unittest.mock import patch, ANY

from regressor_models import (LinearRegressionModel as LRM, 
                              LassoModel as LM, 
                              RidgeModel as RM,
                              ElasticNetModel as ENM,
                              SVRModel as SVRM)
from classifier_models import (LogisticRegressionModel as LoRM,
                               KNNModel as KNNM,
                               SVCModel as SVCM)

from ann_model import MyAnnClass as ANN

from data_processing import DataProcessing as DP
from run_ui import RunUI

class TestDumpModel(unittest.TestCase):

     # Lägg till fler inmatningar här
    def test_dump_model(self):
        self.run_ui = RunUI()
        self.run_ui.csv_file = DP("Adv.csv") 
        self.run_ui.r_or_c = "r"  # Simulera regressormodus
        self.run_ui.df = self.run_ui.csv_file.read_csv()
        self.X, self.y = self.run_ui.csv_file.prepare_for_ml(3)

        # Start ML-processen och dumpa den bästa modellen
        self.run_ui.start_ml()
        with patch("builtins.input", side_effect=["y", "jang"]):
            self.run_ui.dump_best_model()
