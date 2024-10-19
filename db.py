import psycopg2
from psycopg2 import sql, extras
import sys

def setup_db():
    try:
        # Establish a connection to the PostgreSQL database
        connection = psycopg2.connect(
            user='tapasmohanty',
            password='postgres',
            host='localhost',
            port='5432',
            database='face'
        )
        cursor = connection.cursor()

        # Create the cube extension if it doesn't already exist
        cursor.execute("CREATE EXTENSION IF NOT EXISTS cube;")

        # Drop the 'vectors' table if it exists and create a new one
        cursor.execute("DROP TABLE IF EXISTS vectors;")
        cursor.execute("""
            CREATE TABLE vectors (
                id SERIAL PRIMARY KEY,
                file VARCHAR,
                vec_low CUBE,
                vec_high CUBE
            );
        """)

        # Create an index for the 'vectors' table to speed up search operations
        cursor.execute("CREATE INDEX vectors_vec_idx ON vectors (vec_low, vec_high);")

        # Commit the transaction
        connection.commit()

        print("Database setup completed successfully.")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error setting up the database: {error}")
        sys.exit(1)

    finally:
        # Close the cursor and connection to clean up
        if cursor:
            cursor.close()
        if connection:
            connection.close()



setup_db()
