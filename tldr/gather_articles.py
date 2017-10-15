import os
import subprocess
import re
import pickle

# ---------------------------

# get all filenames in directory

cnn_files = ["./cnn/stories/"+f for f in os.listdir("./cnn/stories") if os.path.isfile(os.path.join("./cnn/stories", f))]
dm_files = ["./dailymail/stories/"+f for f in os.listdir("./dailymail/stories")[1:] if os.path.isfile(os.path.join("./dailymail/stories", f))]
all_files = cnn_files + dm_files


# ---------------------------

# helper functions to read story files

def read_text_file(story_file):
    """
    Reads in story file line by line. Story files could not be opened directly and needed to be .
    story_file: story file
    """
    cmd = "cat {} > output.txt".format(story_file)
    os.system(cmd)
    
    lines = []
    with open("output.txt", "r") as file:
        for line in file:
            lines.append(line.strip())
    file.close()
    return lines


def get_highlight(lines, length):
    """
    Combine lines that make up the highlights.
    length: indicates if only want first highlight sentence or all
    """
    combined = []
    for line in lines:
        if line=="@highlight":
            continue
        elif len(line)<=1:
            continue
        else:
            re.sub(r"new: ", "", line)
            combined.append(line.strip())

            # add period when not available
            if length==1:        
                return combined[0] + "."

    return ". ".join(combined) + "."


def get_article(lines):
    """
    Combine lines that make up the article.
    """
    dm_single_close_quote = u'\u2019' # unicode
    dm_double_close_quote = u'\u201d'
    punctuation = [".", "!", "?", "\"", "\'", ")", dm_single_close_quote, dm_double_close_quote]

    combined = []
    # add punctuation when not available
    for line in lines:
        if line[-1] not in punctuation:
            line += ["."]
        combined.append(line.strip())
    return ". ".join(combined) + "."


def create_dict(filepath):
    """
    Create an individual dictionary for each story file with the filename, article, and highlights/summary.
    """
    article_dict = {}
    text = read_text_file(filepath)
    summary_start = text.index("@highlight")
    
    article_dict['file'] = filepath
    article_dict['article'] = " ".join(text[:summary_start]).strip()
    article_dict['highlights'] = get_highlight(text[summary_start:],"all") # retrieve only first highlight
    
    return article_dict


# ---------------------------

# gather articles

all_articles_list = []

for index,file in enumerate(all_files):
    all_articles_list.append(create_dict(file))
    
    # print update of progress
    if index%1000==0:
        print("Working on it. Currently at {}".format(index))

print(len(all_articles_list))

# pickle for future use
pickle.dump(all_articles_list, open("all_articles_list.pkl", "wb"))

