import postgresql
import postgresql.driver as pg_driver

def setup_db():
    db = pg_driver.connect(user = 'postgres',password = 'postgres', host = 'localhost', port = 5432, database = 'face')
    db.execute("create extension if not exists cube;")
    db.execute("drop table if exists vectors")
    db.execute("create table vectors (id serial, file varchar, vec_low cube, vec_high cube);")
    db.execute("create index vectors_vec_idx on vectors (vec_low, vec_high);")


setup_db()
