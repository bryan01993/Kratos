shakespeare = '‘All the world is a stage, and all the men and women merely players. They have their exits and their entrances, And one man in his time plays many parts.’'
#(Tip: the three first words should be ‘ a all all’, this time duplicates are allowed and remember that there are words in mayus)
def sort_string(shakespeare):
    shakespeare = shakespeare.lower()
    shakespeare = shakespeare.replace(['‘','’','.'],' ')
    #shakespeare = shakespeare.replace('’', ' ')
    #shakespeare = shakespeare.replace('.', ' ')
    shakespeare = shakespeare.split(' ')
    shakespeare = sorted(shakespeare)
    shakespeare = shakespeare[4:]
    print('sorted',shakespeare)
    return shakespeare

sort_string(shakespeare)