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
    
def detach_brackets(s):
    return re.sub(r'\([^\)]*\)','',s)

def detach_html_tag(s):
    return re.sub(r'</?[^>]+>','',s)


rule2fuc = {
    'contain_any_url':contain_any_url,
    'contain_any_javadoc_tag':contain_any_javadoc_tag,
    'contain_any_non_English':contain_any_non_English,
    'not_contain_any_letter':not_contain_any_letter,
    'less_than_three_words':less_than_three_words,
    'end_with_question_mark':end_with_question_mark,
    'contain_html_tag':contain_html_tag,
    'detach_brackets':detach_brackets,
    'detach_html_tag':detach_html_tag,
}


def rule_filter(comments,selected_rules=[],defined_rule_dict={}):
    if not isinstance(comments, list):
        raise TypeError('comments must be a list')

    new_comments = []
    indexs = []
    rule2fuc.update(defined_rule_dict)

    if len(selected_rules) < 1:
        selected_rules = list(rule2fuc.keys())
    rule_set = selected_rules + list(defined_rule_dict.keys())

    for i,c in enumerate(comments):
        flag = False
        for rule in rule_set:
            if rule.startswith('detach'):
                result = rule2fuc[rule](c)
                if result is True:
                    continue
                elif result is False:
                    flag = True
                    break
                elif isinstance(result,str):
                    c = result
                else:
                    raise TypeError('Function must return True, False or String')

        for rule in rule_set:
            if not rule.startswith('detach') and rule2fuc[rule](c):
                flag = True
                break

        if not flag:
            new_comments.append(c)
            indexs.append(i)

    return new_comments,indexs
        
