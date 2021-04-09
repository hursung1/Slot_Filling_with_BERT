
from preprocess.gen_embeddings_for_slu import slot_list

import numpy as np
import logging
logger = logging.getLogger()

y1_set = ["O", "B", "I"]
y2_set = ['O', 'B-playlist', 'I-playlist', 'B-music_item', 'I-music_item', 'B-geographic_poi', 'I-geographic_poi', 'B-facility', 'I-facility', 'B-movie_name', 'I-movie_name', 'B-location_name', 'I-location_name', 'B-restaurant_name', 'I-restaurant_name', 'B-track', 'I-track', 'B-restaurant_type', 'I-restaurant_type', 'B-object_part_of_series_type', 'I-object_part_of_series_type', 'B-country', 'I-country', 'B-service', 'I-service', 'B-poi', 'I-poi', 'B-party_size_description', 'I-party_size_description', 'B-served_dish', 'I-served_dish', 'B-genre',  'I-genre', 'B-current_location', 'I-current_location', 'B-object_select', 'I-object_select', 'B-album', 'I-album', 'B-object_name', 'I-object_name', 'B-state', 'I-state', 'B-sort', 'I-sort', 'B-object_location_type', 'I-object_location_type', 'B-movie_type', 'I-movie_type', 'B-spatial_relation', 'I-spatial_relation', 'B-artist', 'I-artist', 'B-cuisine', 'I-cuisine', 'B-entity_name', 'I-entity_name', 'B-object_type', 'I-object_type', 'B-playlist_owner', 'I-playlist_owner', 'B-timeRange', 'I-timeRange', 'B-city', 'I-city', 'B-rating_value', 'B-best_rating', 'B-rating_unit', 'B-year', 'B-party_size_number', 'B-condition_description', 'B-condition_temperature']
domain_set = ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]

SLOT_PAD = 0
PAD_INDEX = 0
UNK_INDEX = 1

class Vocab():
    def __init__(self):
        self.word2index = {"PAD":PAD_INDEX, "UNK":UNK_INDEX}
        self.word2count = {}
        self.index2word = {PAD_INDEX: "PAD", UNK_INDEX: "UNK"}
        self.n_words = 2
    def index_words(self, sentence):
        for word in sentence:
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.word2count[word] = 1
                self.n_words+=1
            else:
                self.word2count[word]+=1

def read_file(filepath, vocab, domain=None):
    domain_utter_list, y1_list, y2_list = [], [], []
    # domain_utter_list: lists of domain concatenated with tokens of query
    # y1_list: lists of BIO labels w/o domain
    # y2_list: lists of BIO labels with domain
    # dm_list: lists of domain
    max_length = 0
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()  # query \t labels
            splits = line.split("\t")
            tokens = splits[0].split()
            l2_list = splits[1].split() # O DM1-B DM1-I ....
            if max_length < len(tokens):
                max_length = len(tokens)

            # tokens.insert(0, domain)
            domain_utter_list.append([domain, tokens])
            y2_list.append(l2_list)

            # update vocab
            vocab.index_words(tokens)

            l1_list = []
            for l in l2_list:
                if "B" in l:
                    l1_list.append("B")
                elif "I" in l:
                    l1_list.append("I")
                else:
                    l1_list.append("O")
            y1_list.append(l1_list)

    data_dict = {"domain_utter": domain_utter_list, "y1": y1_list, "y2": y2_list}
    
    return data_dict, vocab, max_length


def datareader():
    logger.info("Loading and processing data ...")

    data = {"AddToPlaylist": {}, "BookRestaurant": {}, "GetWeather": {}, "PlayMusic": {}, "RateBook": {}, "SearchCreativeWork": {}, "SearchScreeningEvent": {}}
    max_length = {"AddToPlaylist": 0, "BookRestaurant": 0, "GetWeather": 0, "PlayMusic": 0, "RateBook": 0, "SearchCreativeWork": 0, "SearchScreeningEvent": 0}
    # load data and build vocab
    vocab = Vocab()
    data["AddToPlaylist"], vocab, max_length['AddToPlaylist'] = read_file("data/AddToPlaylist/AddToPlaylist.txt", vocab, domain="AddToPlaylist")
    data["BookRestaurant"], vocab, max_length['BookRestaurant'] = read_file("data/BookRestaurant/BookRestaurant.txt", vocab, domain="BookRestaurant")
    data["GetWeather"], vocab, max_length['GetWeather'] = read_file("data/GetWeather/GetWeather.txt", vocab, domain="GetWeather")
    data["PlayMusic"], vocab, max_length['PlayMusic'] = read_file("data/PlayMusic/PlayMusic.txt", vocab, domain="PlayMusic")
    data["RateBook"], vocab, max_length['RateBook'] = read_file("data/RateBook/RateBook.txt", vocab, domain="RateBook")
    data["SearchCreativeWork"], vocab, max_length['SearchCreativeWork'] = read_file("data/SearchCreativeWork/SearchCreativeWork.txt", vocab, domain="SearchCreativeWork")
    data["SearchScreeningEvent"], vocab, max_length['SearchScreeningEvent'] = read_file("data/SearchScreeningEvent/SearchScreeningEvent.txt", vocab, domain="SearchScreeningEvent")

    return data, vocab, max(max_length.values()) 