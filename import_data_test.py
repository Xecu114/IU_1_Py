import unittest

from sqlalchemy import LABEL_STYLE_DISAMBIGUATE_ONLY
import import_data as f

import os

path_for_db = "Python_Course_IU\\my_db.db"
if os.path.exists(path_for_db):
    os.remove(path_for_db)

# eine Testklasse definieren
# die die TestCases von unittest erbt


class UnitTestMathOperations(unittest.TestCase):

    def test_func(self):
        '''
        '''
        result = f.import_train_csv_to_sqllite_table()
        self.assertIsInstance(result, list)

        comp_item_first = "train_data_row(x=-20.0, y1=100.216064,"\
            " y2=-19.757296, y3=0.3461139, y4=19.776287)"
        self.assertEqual(str(result[0]), str(comp_item_first))
        comp_item_last = "train_data_row(x=19.9, y1=99.1435,"\
            " y2=20.025005, y3=0.102107115, y4=19.580418)"
        self.assertEqual(str(result[-1]), comp_item_last)


# dieses Skript im unittest-Kontext ausführen
if __name__ == '__main__':
    unittest.main()
