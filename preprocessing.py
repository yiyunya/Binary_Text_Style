from numpy import *

__all__ = [
    'Preprocessing'
]

class Preprocessing:



    @staticmethod
    def load_data(option="train"):
        print("Loading "+option+" data...")
        f = open("data/"+option+".in")
        print("Loading text...")
        content_src = f.readlines()
        f.close()
        f = open("data/"+option+".out")
        print("Loading categories...")
        cate_src = f.readlines()
        f.close()
        data = [content_src, cate_src]

        print("Data loading done.")

        return data


def load_test_data():
    return Preprocessing.load_data(option="test")


def load_train_data():
    return Preprocessing.load_data()



