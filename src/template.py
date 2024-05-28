LABEL_MAP = {
    'agnews':{'Business':'Business', 'Technology':'Technology', 'World':'Technology', 'Sports':'Technology'},
    'boolq':{True:'True',False:'False'},
    'cb':{'entailment':'yes', 'neutral':'maybe', 'contradiction':'no'},
    'dbpedia':{'Album':'Album',
                'Animal':'Animal',
                'Artist':'Artist',
                'Athlete':'Athlete',
                'Book':'Book',
                'Building':'Building',
                'Company':'Company',
                'Film':'Film',
                'Nature':'Nature',
                'Plant':'Plant',
                'Politician':'Politician',
                'School':'School',
                'Transportation':'Transportation',
                'Village':'Village'},
    'imdb':{0: 'negative', 1: 'positive'},    
    'mnli':{'entailment':'yes', 'neutral':'maybe', 'contradiction':'no'}, 
    'rte':{'entailment':'yes', 'not_entailment':'no'}, 
    'scicite':{0:'method', 1:'result', 2:'background'},   
    'sst5':{0:'terrible', 1:'bad', 2:'okay', 3:'good', 4:'great'},
    'subj':{'objective':'objective', 'subjective':'subjective'}, 
    'trec':{'Abbreviation':'Abbreviation',
            'Description': 'Description',
            'Entity':'Entity',
            'Location': 'Location',
            'Number':'Number',
            'Person':'Person'},   
    'yahoo':{ 0:'Society',
              1:'Science', 
              2:'Health', 
              3:'Education', 
              4:'Computers', 
              5:'Sports', 
              6:'Business', 
              7:'Music', 
              8:'Family', 
              9:'Politics'},     
}

# TEMPLATE = {
#     'agnews':"Article:[sentence]\nTopic:[label]",
#     'boolq':"Passage:[passage]\nQuestion:[question]\nAnswer:[label]",
#     'cb':"Premise:[premise]\nHypothesis:[hypothesis]\nLabel:[label]",
#     'dbpedia':"Article:[sentence]\nTopic:[label]",
#     'imdb':"Review:[text]\nSentiment:[label]",    
#     'mnli':"Premise:[premise]\nHypothesis:[hypothesis]\nLabel:[label]",
#     'rte':"Premise:[premise]\nHypothesis:[hypothesis]\nLabel:[label]",
#     'scicite':"Citation:[text]\nAnswer:[label]",   
#     'sst5':"Review:[text]\nSentiment:[label]",
#     'subj':"Sentence:[sentence]\nIt is [label]",
#     'trec':"Question:[sentence]\nTopic:[label]",
#     'yahoo':"Article:[question_title] [question_content] [best_answer]\nTopic:[label]",   
# }

# HEAD = {
#     'agnews':"Classify the topic of the article.",
#     'boolq':"Read the passage and answer the question by yes or no.", 
#     'cb':"Determine if the hypothesis is true based on the premise.",
#     'dbpedia':"Classify the topic of the article.",
#     'imdb':"Classify the sentiment of the review.",    
#     'mnli':"Determine if the hypothesis is true based on the premise.",
#     'rte':"Determine if the hypothesis is true based on the premise.",
#     'scicite':"Is the following citation from a scientific paper describing a method, a result, or background?",  
#     'sst5':"Classify the sentiment of the review.",    
#     'subj':"Determine the subjectivity of the sentence.",
#     'trec':"Classify the topic of the question.",
#     'yahoo':"Classify the topic of the article.",
# }


HEAD = {
    'sst2':"Classify the reviews based on whether their sentiment type is positive or negative.",
    'rte':"Classify the pairs of premise and hypothesis based on whether the relationship type is yes or no.",
    'mnli':"Classify the pairs of premise and hypothesis based on whether the relationship type is yes, no or maybe.",
    'qnli':'Classify whether the sentence contains the answer to the question. The answer is yes or no.',
    'mrpc':"Classify whether the two sentences are semantically equivalent. The answer is yes or no.",     
    'cola':"Determine if the sentence is grammatically correct. The answer is yes or no.", 
}

TEMPLATE = {
    'sst2':"Review:[sentence]\nSentiment:\n[label]",
    'rte':"Premise:[premise]\nHypothesis:[hypothesis]\nLabel:\n[label]",
    'mnli':"Premise:[premise]\nHypothesis:[hypothesis]\nLabel:\n[label]",
    'qnli':"Sentence:[sentence]\nQuestion:[question]\nLabel:\n[label]",
    'mrpc':"Sentence1:[sentence1]\nSentence2:[sentence2]\nLabel:\n[label]",
    'cola':"Sentence:[sentence]\nLabel:\n[label]",  
}

LABEL = {
    'sst2':['negative','positive'],
    'rte':['yes','no'],
    'mnli':['yes','no','maybe'],
    'qnli':['yes','no'],
    'mrpc':['yes','no'],
    'cola':['yes','no'],
}

# SENTENCE_HEAD = {
#     'sst2':"Review:",
#     'rte':"Premise:[premise]\nHypothesis:[hypothesis]",
#     'mnli':"Premise:[premise]\nHypothesis:[hypothesis]",
#     'qnli':"Sentence:[sentence]\nQuestion:[question]",
#     'mrpc':"Sentence1:[sentence1]\nSentence2:[sentence2]",
#     'cola':"Sentence:[sentence]",  
# }

# LABEL_HEAD = {
#     'sst2':"Review:[sentence]\nSentiment:\n[label]",
#     'rte':"Premise:[premise]\nHypothesis:[hypothesis]\nLabel:\n[label]",
#     'mnli':"Premise:[premise]\nHypothesis:[hypothesis]\nLabel:\n[label]",
#     'qnli':"Sentence:[sentence]\nQuestion:[question]\nLabel:\n[label]",
#     'mrpc':"Sentence1:[sentence1]\nSentence2:[sentence2]\nLabel:\n[label]",
#     'cola':"Sentence:[sentence]\nLabel:\n[label]",  
# }