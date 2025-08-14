import pandas as pd
from sqlalchemy import create_engine
import traceback
import os


class QueryDatabase: 

    def __init__(self):
        self.db_engine = create_engine(os.getenv("DB_CONN_STRING"))
        print("Built DB engine")

    def write_data(self, df, table_name): 

        try: 
            print(f"Attempting to write {len(df)} rows to {table_name}")

            with self.db_engine.connect() as conn: 
                df.to_sql(name=table_name, schema='train', con=conn, if_exists='append', index=False, chunksize=1000, method='multi')

            print(f"Sucessfully wrote data to {table_name}")

            return self

        except Exception: 
            print(f"Failed to write data to {table_name}")
            print(f"Full traceback: {traceback.format_exc()}")
            return None
        
        finally: 
            self.db_engine.dispose()

        
    def read_data(self, query):

        try: 
            print("Attempting to read data")
            with self.db_engine.connect() as conn: 
                df = pd.read_sql_query(sql=query, con=conn)

            print("Sucessfully read data")

            return df
        
        except Exception as e:
            print(f"Failed to read data: {e}")
            print(f"Full traceback: {traceback.format_exc()}")
            return None 

        finally: 
            self.db_engine.dispose()
