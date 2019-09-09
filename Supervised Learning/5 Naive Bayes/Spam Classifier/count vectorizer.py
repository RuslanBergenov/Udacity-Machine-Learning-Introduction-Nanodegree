# =============================================================================
# Implementing Bag of Words in scikit-learn
# =============================================================================
'''
Solution
'''
from sklearn.feature_extraction.text import CountVectorizer
count_vector_1 = CountVectorizer(stop_words='english')



# =============================================================================
# Data preprocessing with CountVectorizer()
# =============================================================================
print(count_vector_1)




'''
Solution:
'''
# No need to revise this code
count_vector_1.fit(df['sms_message'])
count_vector_1.get_feature_names()



'''
Solution
'''
doc_array_1 = count_vector_1.transform(df['sms_message']).toarray()



'''
Solution
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
'''
frequency_matrix_1 = pd.DataFrame(doc_array_1, columns = count_vector_1.get_feature_names())
frequency_matrix_1