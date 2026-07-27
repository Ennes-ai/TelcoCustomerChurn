import unittest
import pandas as pd
import prepare


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.DataFrame(pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv'))
        cls.prepare = prepare.Prepare()

    def test_colums(self):
        transformed_df = self.prepare.transform(self.data)

        self.assertNotIn("Musteri_ID" , transformed_df.columns)
        self.assertNotIn("customerID" , transformed_df.columns)
        self.assertNotIn("Toplam_Ucret" , transformed_df.columns)

    def test_no_string_column(self):
        transformed_df = self.prepare.transform(self.data)

        string_column = transformed_df.select_dtypes(include=['object','str'])

        self.assertEqual(len(string_column.columns), 0)

    def test_transform_return(self):
        transformed_df = self.prepare.transform(self.data)

        self.assertEqual(first= type(self.data) , second= type(transformed_df) , msg= "Wrong DataFrame type")

if __name__ == '__main__':
    unittest.main()
