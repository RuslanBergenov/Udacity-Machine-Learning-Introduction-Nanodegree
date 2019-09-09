# =============================================================================
# IMPORT
# =============================================================================
import pandas as pd
# Dataset available using filepath 'smsspamcollection/SMSSpamCollection'
df = pd.read_table('smsspamcollection/SMSSpamCollection', sep = '\t', names = ['label', 'sms_message'])

# Output printing out first 5 rows
df.head()

df.info()

df['label'] = df.label.map({'ham':0, 'spam':1})


df.head()



# =============================================================================
# Step 1: Convert all strings to their lower case form.
# =============================================================================
'''
Solution:
'''
lower_case_documents_1 = []
for i in df['sms_message']:
    i = i.lower()
    lower_case_documents_1.append(i)
print(lower_case_documents_1)


# =============================================================================
# Step 2: Removing all punctuation
# =============================================================================
'''
Solution:
'''
sans_punctuation_documents_1 = []
import string

for i in lower_case_documents_1:
    # https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
    i = i.translate(str.maketrans('', '', string.punctuation))
    sans_punctuation_documents_1.append(i)
    
print(sans_punctuation_documents_1)






# =============================================================================
# Step 3: Tokenization
# =============================================================================
'''
Solution:
'''
# https://stackoverflow.com/questions/1546226/simple-way-to-remove-multiple-spaces-in-a-string
# https://stackoverflow.com/questions/7899525/how-to-split-a-string-by-space
import re
preprocessed_documents_1 = []
for i in sans_punctuation_documents_1:
    i = re.sub(' +', ' ', i).split(' ')
    preprocessed_documents_1.append(i)
print(preprocessed_documents_1)







# =============================================================================
# Step 4: Count frequencies
# =============================================================================
'''
Solution
'''
frequency_list_1 = []
import pprint
from collections import Counter

for i in preprocessed_documents_1:
    myCounter = Counter(i)
    frequency_list_1.append(myCounter)
    
pprint.pprint(frequency_list_1)






