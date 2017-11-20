import numpy as np


class Count(object):
    def __init__(self, ham_count=0, spam_count=0):
        self.ham_count = ham_count
        self.spam_count = spam_count

    def total(self):
        return self.ham_count + self.spam_count

    def __repr__(self):
        return self.__dict__.__repr__()

class FeatureProbability:
    def __init__(self):
        self.class_count = Count()
        self.code_count = dict()

    @classmethod
    def from_email_set(cls, email_set):
        feature_prob_set = cls()

        for ham_email in email_set.ham_emails:
            feature_prob_set.add_email(ham_email, True)
        for spam_email in email_set.spam_emails:
            feature_prob_set.add_email(spam_email, False)

        return feature_prob_set

    def add_email(self, email, is_ham_email):
        if is_ham_email:
            self.class_count.ham_count += 1
        else:
            self.class_count.spam_count += 1

        for code in email.codes:
            if code not in self.code_count:
                self.code_count[code] = Count()

            if is_ham_email:
                self.code_count[code].ham_count += 1
            else:
                self.code_count[code].spam_count += 1

    def code_prob_ratio(self, code):
        code_count = self.code_count[code]

        # Calculate conditional probabilities
        code_given_spam_prob = float(code_count.spam_count) / self.class_count.spam_count

        # What if ham count is zero?
        if code_count.ham_count == 0:
            return np.inf

        code_given_ham_prob = float(code_count.ham_count) / self.class_count.ham_count

        return code_given_spam_prob / code_given_ham_prob
