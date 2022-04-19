import nltk
from nltk.corpus import wordnet

import re
import fasttext
import numpy as np
from tensorflow import keras



class IngredientExtractor():
    
    def __init__(self, keras_model_path):
        """
            Init script and load models.
        """
        self.ft_model = fasttext.load_model('cc.en.300.bin')
        self.model = keras.models.load_model(keras_model_path)
        

        
    def recipe_to_target_words(self, recipe):
        """
            Create word list from recipe. Include 2-gram and 3-grams.
        """
        words = nltk.word_tokenize(recipe)
        #Create vector for each word
        word_vectors = np.array([self.ft_model.get_word_vector(word) for word in words])

        targets = []
        for i in range(len(word_vectors)):    

            # Calculate mean vector for n-grams.
            n_gram_1 = word_vectors[i:i+1].mean(axis=0)
            n_gram_2 = word_vectors[i:i+2].mean(axis=0)    
            n_gram_3 = word_vectors[i:i+3].mean(axis=0)

            targets.append((words[i], n_gram_1))
            targets.append((' '.join(words[i:i+2]), n_gram_2))
            targets.append((' '.join(words[i:i+3]), n_gram_3))

        return targets

    def predict(self, text):
        """
            Prediction function

            Parameters
            ----------
            text : str
                String to create prediction
                
            Returns
            -------
            [str]
                List of ingredient words
        """
        # Get target word list
        targets = self.recipe_to_target_words(text)

        # Seperate words and vectors
        words = [t[0] for t in targets]
        vecs = np.array([np.expand_dims(t[1],axis=1) for t in targets])

        # Create prediciton for each word or n-gram
        y_pred = self.model.predict(vecs) 
        y_pred = np.squeeze(y_pred > 0.5).astype(int)

        result = []
        for w,p in zip(words,y_pred):
            # if prediction is 1 --> Ingredient. add to list
            if p==1:
                result.append(w)

        return result

    def post_filter(self, items):    
        """
            Post filtering function of n-gram results. to have clear text

            Parameters
            ----------
            items : [str]
                List of words predicted with keras model.
                
            Returns
            -------
            [str]
                Filtered list of words.
        """

        # Seperate multiple words by comma & "and" word
        filtered = []
        for f in items:
            if ',' in f:
                filtered += f.split(',')
            if 'and' in f:
                filtered += f.split('and')
            if 'and' not in f and ',' not in f:
                filtered.append(f)

        filtered = list(set(filtered))        
        filtered = [f.strip() for f in filtered if len(f) > 0]

        # SINGULARIZE plural words
        filtered = [f if wordnet.morphy(f) is None else wordnet.morphy(f) for f in filtered]    

        # filter duplicate words by n-grams
        filtered = sorted(filtered, key=lambda x: len(x.split()), reverse=True)    


        # remove single word of "cup"
        ignore_items = ["cup"]
        filtered = list(set(filtered) - set(ignore_items))   

        return filtered

    def find_ingredients_single_text(self, text):
        """
            Returns ingredient list for single text

            Parameters
            ----------
            text : str
                String to find ingredients
                
            Returns
            -------
            [(word: str,start_index: int,end_index: int)]
                Filtered list of ingredients and start end poses.
        """
        ingredients = self.post_filter(self.predict(text))

        matches = []
        for item in ingredients:
            matches += [[item, match.start(),match.end()] for match in re.finditer(item, text)]

        return matches

    def create_result_from_object(self, json):
        """
            Creates prediction output for given json.

            Parameters
            ----------
            json : dict
                Dictionary of recipies. e.g. { "recipe1": "","recipe2": ""}
                
            Returns
            -------
            dict
                Dictionary with same keys as input with ingredient values.
        """
        result = {}
        for key in json:
            result[key] = self.find_ingredients_single_text(json[key])

        return result