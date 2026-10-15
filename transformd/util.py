'''
util.py

Misc. utility functions
'''


def title_case(sentence):
    '''
    Convert a string to title case.
    Title case means that the first letter of each word is capitalized, and all other letters in the word are lowercase. 

    Parameters
    ____
    sentence: str
        String to be converted to title case
    Returns
    _____
    ret: str
        String converted to title case
    Example
    _____
    >>title_case('ThIS is A sTRiNg to BE ConvERTed.')
    'This Is A String To Be Converted.'
    '''

    ret = sentence[0].upper()

    # Loop through the rest of the characters
    for i in range(1, len(sentence)):
        if sentence[i - 1] == ' ':
            ret = ret + sentence[i].upper()
        else:
            ret = ret + sentence[i].lower()

    return ret
