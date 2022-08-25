import pandas as pd
from bs4 import BeautifulSoup
import requests
import os
import re
import spacy
nlp = spacy.load('en_core_web_sm')

class DownloadCsv : 
    
    def __init__(self) :
        
        print("Wait...")
        # Fine headers User-agent of your current browser
        # https://www.whatismybrowser.com/detect/what-is-my-user-agent/
        self.headers = {
            'User-agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'
        }
        
        
        # df_articles is nothing but contains all data information about all articles
        self.df_articles = pd.DataFrame()

        # The Stop Words Lists (found in the folder StopWords) are used to clean the text so that Sentiment Analysis can be performed by excluding the words found in Stop Words List.
        self.stopwords_set = set()
        try : 
            for itm in os.listdir(r"StopWords/") :
                with open(f'StopWords/{itm}', 'r') as f : 
                    stop = f.read()
                    # print(stop)
                    for wd in stop.split("\n") :
                        first_wd = wd.split(' ')
                        self.stopwords_set.add(first_wd[0].lower())
                        # print(first_wd[0].lower())
            self.stopwords_set.remove('')
            print("Custom stopwords seted....")
        except : 
            print("There must be StopWords/ folder and in that folder have some txt stopwords files...")


        # store positive and negative words in python set
        self.negative_words_set = set()
        try : 
            with open(r"MasterDictionary/negative-words.txt", 'r') as f : 
                nws = f.read()
                # print(stop)
                for wd in nws.split("\n") :
                    self.negative_words_set.add(wd.lower())
                    # print(wd.lower())
                self.negative_words_set.remove('')
            print("Negative words has seted...")
        except : 
            print("There must be MasterDictionary/negative-words.txt negative words file...")


        # store positive and negative words in python set
        self.positive_words_set = set()
        try : 
            with open(r"MasterDictionary/positive-words.txt", 'r') as f : 
                pws = f.read()
                # print(stop)
                for wd in pws.split("\n") :
                    self.positive_words_set.add(wd.lower())
                    # print(wd.lower())
                self.positive_words_set.remove('')
            print("Positive words has seted...")
        except : 
            print("There must be MasterDictionary/positive-words.txt positive words file...")

    # function to return article title and content
    def return_article_title_content(self, url) : 
        try : 
            soup = BeautifulSoup(requests.get(url, headers = self.headers).text, 'lxml')
            return soup.find('h1', class_  = 'entry-title').text, soup.find('div', class_ = 'td-post-content').text
        except : 
            print(f"{url} : Is Invalid url...")

    # Function to clean text    
    def clean_text(self, row) : 
        row = re.sub("\n", "", row).lower()
        return " ".join([itm.lemma_ for itm in nlp(row) if not (itm in self.stopwords_set or itm.is_punct or itm.is_stop or itm.is_space ) ])

    # Functon for POSITIVE SCORE
    def positiveScore(self,row) : 
        ps = 0
        for wd in row['cleanContent'].split(" ") : 
            if wd in self.positive_words_set : 
                ps += 1
        return ps

    # Function for NEGATIVE SCORE
    def negativeScore(self, row) : 
        ns = 0
        for wd in row['cleanContent'].split(" ") : 
            if wd in self.negative_words_set : 
                ns += 1
        return ns

    # Function for POLARITY SCORE
    def polarityScore(self, row) : 
        return (row['POSITIVE SCORE'] - row['NEGATIVE SCORE']) / ((row['POSITIVE SCORE'] + row['NEGATIVE SCORE']) + 0.000001)


    # Function for SUBJECTIVITY SCORE
    def subjectivityScore(self, row) : 
        total_words_after_cleaning = len(row['cleanContent'].split(" "))
        return (row['POSITIVE SCORE'] + row['NEGATIVE SCORE']) / ((total_words_after_cleaning) + 0.000001)


    # Function for AVG SENTENCE LENGTH
    def avgSentLenght(self, row) : 
        row = re.sub("\n", "", row['Content']).lower()
        tot_words = 0
        tot_sents = 0
        doc_row = nlp(row)
        
        for itm in doc_row : 
            if not(itm.is_punct or itm.is_space) : 
                tot_words += 1
        
        for itm in doc_row.sents : 
            tot_sents += 1
        
        return tot_words / ( tot_sents + 0.0001)


    # Function for number of syllables return
    def no_of_syllables(self, word) : 
        syllables_cnt = 0
        for ch in word : 
            if ch in 'aeiou' : 
                syllables_cnt += 1
                
        return syllables_cnt
        
    # Function for PERCENTAGE OF COMPLEX WORDS
    def percentageComplexWords(self, row) : 
        row = re.sub("\n", "", row['Content']).lower()
        tot_words = 0
        tot_complex_words = 0
        
        doc_row = nlp(row)
        
        for itm in doc_row : 
            if not(itm.is_punct or itm.is_space) : 
                tot_words += 1
                        
                if self.no_of_syllables(itm.lemma_) > 2 : 
                    tot_complex_words += 1
        
        return tot_words / ( tot_complex_words + 0.0001)


    # Function for FOG INDEX
    def fogIndex(self, row) : 
        return 0.4 * row['AVG SENTENCE LENGTH'] + row['PERCENTAGE OF COMPLEX WORDS']


    # Function for COMPLEX WORD COUNT
    def complexWordCount(self, row) :
        row = re.sub("\n", "", row['Content']).lower()
        tot_complex_words = 0
        
        doc_row = nlp(row)
        
        for itm in doc_row : 
            if not(itm.is_punct or itm.is_space) :         
                if self.no_of_syllables(itm.lemma_) > 2 : 
                    tot_complex_words += 1
        
        return tot_complex_words


    # Functon for WORD COUNT
    def wordCount(self, row) : 
        return len(row['cleanContent'].split(" "))


    # Function for SYLLABLE PER WORD
    def syllableCountPerWord(self, row) : 
        row = re.sub("\n", "", row['Content']).lower()
        tot_syllable = 0
        tot_words =  0
        doc_row = nlp(row)
        
        for itm in doc_row : 
            if not(itm.is_punct or itm.is_space) :
                tot_words += 1
                tot_syllable += self.no_of_syllables(itm.lemma_) 
        
        return tot_syllable / (tot_words + 0.0001) 

    # Function for PERSONAL PRONOUNS
    def personalPronouns(self, row) : 
        row = re.sub("\n", "", row['Content']).lower()
        tot_personal_pronouns = 0
        pp_list = ['i', 'we', 'my', 'ours', 'us']
        doc_row = nlp(row)
        
        for itm in doc_row : 
            if not( itm.is_punct or itm.is_space) :         
                if itm.text in pp_list : 
                    tot_personal_pronouns += 1
                    
        return tot_personal_pronouns

    # Function for AVG WORD LENGTH
    def averageWordLength(self, row) : 
        row = re.sub("\n", "", row['Content']).lower()
        doc_row = nlp(row)
        
        return sum([len(wd) for wd in doc_row]) / len(doc_row)




    # accept url of ariticles or all url as in data frame
    # initilize dataframe in which URL_ID and URL columns must be there
    def input_url(self, file_url, file = True, ID = 1) : 
    
        if file : 
            try : 
                self.df_url = pd.read_csv(file_url)
                if len(self.df_url['URL_ID']) and len(self.df_url['URL']) :
                    print("Valid csv file...")
            except : 
                print("File is not csv or URL_ID nad URL columns not found in the file...")
        else : 
            try : 
                self.df_url = pd.DataFrame({
                    "URL_ID" : [ID],
                    "URL" : [file_url],
                })
                print(f"{file_url} accepted...")
            except : 
                print("url not valid...")

    
        # this is help to create df_articles
        articles_dic = {
            'URL_ID' : [],
            'URL' : [],
            'Title' : [],
            'Content' : []
        }

        # print(self.df_url)
        try : 
            
            for i in range(self.df_url.shape[0]) : 
            
                # each row of df_output dataFarame
                row = self.df_url.iloc[i,:]
                
                # row['URL'] => url of each article
                # return_article_text return as each Article all content of url row['URL']
                title, content = self.return_article_title_content(row['URL'])
                
                # append data in articles_dic
                articles_dic['URL_ID'].append(row['URL_ID'])
                articles_dic['URL'].append(row['URL'])
                articles_dic['Title'].append(title)
                articles_dic['Content'].append(content)
                
                # to save each article_content as row['URL_ID'] name in txt form in Extracted articles folder
                with open(f"Extracted articles/{row['URL_ID']}.txt", 'w', encoding='utf-8') as f : 
                    f.write(title + " " + content)
                    
                # indicate particular article content downloaded
                print(f"{row['URL']} completed...")
                # break
            
            self.df_articles = pd.DataFrame(articles_dic)

        except : 
            print("There are problems to download file or in urls...")


        # create new column of clean article content
        # remove stopwords, pun, custom stopwords...        
        try : 
            self.df_articles['cleanContent'] = self.df_articles['Content'].apply(self.clean_text)
            print("stopwords, punctuation, custom punctiation removed...")
        except : 
            print("problem to remove stopwords, punctuation, custom punctiation...")

        # POSITIVE SCORE add column
        try : 
            self.df_articles['POSITIVE SCORE'] = self.df_articles.apply(self.positiveScore, axis = 1)
            print("POSITIVE SCORE column added...")
        except : 
            print("To add column of POSITIVE SCORE, there are problems...")
        
        
        # NEGATIVE SCORE add new columns
        try : 
            self.df_articles['NEGATIVE SCORE'] = self.df_articles.apply(self.negativeScore, axis = 1)
            print("NEGATIVE SCORE column added...")
        except : 
            print("To add column of NEGATIVE SCORE, there are problems...")
        

        # POLARITY SCORE add new columns
        try : 
            self.df_articles['POLARITY SCORE'] = self.df_articles.apply(self.polarityScore, axis = 1)
            print("POLARITY SCORE column added...")
        except : 
            print("To add column POLARITY SCORE, there are problems...")


        # SUBJECTIVITY SCORE add new columns
        try : 
            self.df_articles['SUBJECTIVITY SCORE'] = self.df_articles.apply(self.subjectivityScore, axis = 1)
            print("SUBJECTIVITY SCORE column added...")
        except : 
            print("To add column SUBJECTIVITY SCORE, there are problems...")
        
        # AVG SENTENCE LENGTH add new column
        try : 
            self.df_articles['AVG SENTENCE LENGTH'] = self.df_articles.apply(self.avgSentLenght, axis = 1)
            print("AVG SENTENCE LENGTH column added...")
        except : 
            print("To add column AVG SENTENCE LENGTH, there are problems...")
        
        # PERCENTAGE OF COMPLEX WORDS add new column
        try : 
            self.df_articles['PERCENTAGE OF COMPLEX WORDS'] = self.df_articles.apply(self.percentageComplexWords, axis = 1)
            print("PERCENTAGE OF COMPLEX WORDS column added...")
        except : 
            print("To add column PERCENTAGE OF COMPLEX WORDS, there are problems...")
        
        # FOG INDEX add new column
        try : 
            self.df_articles['FOG INDEX'] = self.df_articles.apply(self.fogIndex, axis = 1)
            print("FOG INDEX column added...")
        except : 
            print("To add column FOG INDEX, there are problems...")
        
        # AVG NUMBER OF WORDS PER SENTENCE add new column
        try : 
            self.df_articles['AVG NUMBER OF WORDS PER SENTENCE'] = self.df_articles['AVG SENTENCE LENGTH']
            print("AVG NUMBER OF WORDS PER SENTENCE column added...")
        except : 
            print("To add column AVG NUMBER OF WORDS PER SENTENCE, there are problems...")
        
        # COMPLEX WORD COUNT add new column
        try : 
            self.df_articles['COMPLEX WORD COUNT'] = self.df_articles.apply(self.complexWordCount, axis = 1)
            print("COMPLEX WORD COUNT column added...")
        except : 
            print("To add column COMPLEX WORD COUNT, there are problems...")
        
        # WORD COUNT add new column
        try : 
            self.df_articles['WORD COUNT'] = self.df_articles.apply(self.wordCount, axis = 1)
            print("WORD COUNT column added...")
        except : 
            print("To add column WORD COUNT, there are problems...")
        
        # SYLLABLE PER WORD add new column
        try : 
            self.df_articles['SYLLABLE PER WORD'] = self.df_articles.apply(self.syllableCountPerWord, axis = 1)
            print("SYLLABLE PER WORD column added...")
        except : 
            print("To add column SYLLABLE PER WORD, there are problems...")
        
        # PERSONAL PRONOUNS add new column
        try : 
            self.df_articles['PERSONAL PRONOUNS'] = self.df_articles.apply(self.personalPronouns, axis = 1)
            print("PERSONAL PRONOUNS column added...")
        except : 
            print("To add column PERSONAL PRONOUNS, there are problems...")
        
        # AVG WORD LENGTH add new column
        try : 
            self.df_articles['AVG WORD LENGTH'] = self.df_articles.apply(self.averageWordLength, axis = 1)
            print("AVG WORD LENGTH column added...")
        except : 
            print("To add column AVG WORD LENGTH, there are problems...")

        # Save final output in current folder 
        try : 
            self.df_articles[['URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT', 'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH']].to_csv('output.csv', index=False)
            print("Finally output file name as output.csv saved in current folder...")
        except : 
            print("There are problems to save final out name as output.csv...")

        ########## completed ##########


# main function 
if __name__ == "__main__" : 
    
    download_obj = DownloadCsv()
    check = input('Enter 1 for url or other for file : ')
    if check == "1" : 
        file_url = input('Enter valid article url\n')
        ID = input("Enter text file name which will save\n")
        download_obj.input_url(file_url, file=False, ID = ID)
    else : 
        file_url = input('Enter csv file path,it might be URL_ID and URL columns in that csv file\n')
        download_obj.input_url(file_url, file=True)
        