import numpy as np
import os
import os.path
import pickle
from email import Email
from word_encoding_dictionary import WordEncodingDictionary


DATA_DIR = os.path.join(os.getcwd(), "../data/")
ENRON_DATA_DIR_NAME = "enron1"

class EmailSet(object):
    def __init__(self, word_encoding_dictionary, ham_emails, spam_emails):
        self.word_encoding_dictionary = word_encoding_dictionary
        self.ham_emails = ham_emails
        self.spam_emails = spam_emails

    INSTANCE = None
    @classmethod
    def get(cls):
        if not cls.INSTANCE:
            with open(os.path.join(DATA_DIR, 'data.p'), 'rb') as f:
                cls.INSTANCE = pickle.load(f)
        return cls.INSTANCE


def read_email_dir(word_encoding_dictionary, path, label):
    emails = []
    for email_fname in os.listdir(os.path.join(DATA_DIR, path)):
        email_path = os.path.join(path, email_fname)
        email = Email.read(
            path = email_path,
            word_encoding_dictionary = word_encoding_dictionary,
            label = label
        )
        emails.append(email)

    return emails


def build_email_set():
    word_encoding_dictionary = WordEncodingDictionary()
    ham_emails = read_email_dir(
        word_encoding_dictionary=word_encoding_dictionary,
        path=os.path.join(ENRON_DATA_DIR_NAME, "ham"),
        label=0
    )
    spam_emails = read_email_dir(
        word_encoding_dictionary=word_encoding_dictionary,
        path=os.path.join(ENRON_DATA_DIR_NAME, "spam"),
        label=1
    )

    return EmailSet(
        word_encoding_dictionary=word_encoding_dictionary,
        ham_emails=ham_emails,
        spam_emails=spam_emails
    )


def save_email_set(email_set):
    with open("../data/emails.p", "wb") as f:
        pickle.dump(email_set, f)


def build_and_save_email_set():
    if os.path.isfile("../data/emails.p"):
        print "Dataset already processed!"
        return

    print "Reading and processing emails!"
    email_set = build_email_set()
    save_email_set(email_set)
    print "Email set is created!"


if __name__ == "__main__":
    build_and_save_email_set()
