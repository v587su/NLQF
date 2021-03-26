import re


def contain_any_url(s): 
    return bool(re.search(r'://',s))


def contain_any_javadoc_tag(s): 
    return bool(re.search(r'@[a-zA-Z0-9]+',s))
    

def contain_any_non_English(s): 
    return bool(re.search(r'[^\x00-\xff]',s))


def not_contain_any_letter(s): 
    return not bool(re.search(r'[a-zA-Z]', s))


def less_than_three_words(s): 
    words = s.split()
    if (len(words)<=2):
        return True
    else:
        return False
    

def end_with_question_mark(s):
    if len(s) < 1:
        return False
    return s[-1] == '?'

def contain_html_tag(s): 
    return bool(re.search(r'</?[^>]+>', s))
    
def remove_brackets(s):
    return re.sub(r'\([^\)]*\)','',s)

def remove_html_tag(s):
    return re.sub(r'</?[^>]+>','',s)


rule2fuc = {
    'contain_any_url':contain_any_url,
    'contain_any_javadoc_tag':contain_any_javadoc_tag,
    'contain_any_non_English':contain_any_non_English,
    'not_contain_any_letter':not_contain_any_letter,
    'less_than_three_words':less_than_three_words,
    'end_with_question_mark':end_with_question_mark,
    'contain_html_tag':contain_html_tag,
    'remove_brackets':remove_brackets,
    'remove_html_tag':remove_html_tag,
}


def rule_filter(comments,rule_set=[],rule_dic={}):
    if not isinstance(comments, list):
        raise TypeError('comments must be a list')

    new_comments = []
    indexs = []
    rule2fuc.update(rule_dic)

    if len(rule_set) < 1:
        rule_set = rule2fuc.keys()

    for i,c in enumerate(comments):
        for rule in rule_set:
            if rule.startswith('remove'):
                c = rule2fuc[rule](c)

        flag = False
        for rule in rule_set:
            if not rule.startswith('remove') and rule2fuc[rule](c):
                flag = True
                break

        if not flag:
            new_comments.append(c)
            indexs.append(i)

    return new_comments,indexs
        
