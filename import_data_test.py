import unittest
import import_data as f
import os


dir_path = "Python_Course_IU"
if os.path.exists(dir_path+"\\my_db.db"):
    os.remove(dir_path+"\\my_db.db")


# eine Testklasse definieren
# die die TestCases von unittest erbt
class UnitTestMathOperations(unittest.TestCase):

    def test_func(self):
        '''
        '''
        train_result, ideal_result = f.import_datasets_to_sqllite_table(
            dir_path)
        self.assertIsInstance(train_result, list)
        self.assertIsInstance(ideal_result, list)

        train_comp_item_first = "train_data(x=-20.0, y1=100.216064,"\
            " y2=-19.757296, y3=0.3461139, y4=19.776287)"
        train_comp_item_last = "train_data(x=19.9, y1=99.1435,"\
            " y2=20.025005, y3=0.102107115, y4=19.580418)"
        self.assertEqual(str(train_result[0]), train_comp_item_first)
        self.assertEqual(str(train_result[-1]), train_comp_item_last)


# dieses Skript im unittest-Kontext ausführen
if __name__ == '__main__':
    unittest.main()
