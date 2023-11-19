import unittest
import import_train_data as f
import sqlalchemy.orm


# eine Testklasse definieren
# die die TestCases von unittest erbt


class UnitTestMathOperations(unittest.TestCase):

    def test_create_sqllite_table(self):
        '''
        Funktion zum Testen von create_sqllite_table
        '''
        result = f.create_sqllite_table()
        self.assertIsInstance(result, sqlalchemy.orm.decl_api.DeclarativeMeta)


# dieses Skript im unittest-Kontext ausführen
if __name__ == '__main__':
    unittest.main()
