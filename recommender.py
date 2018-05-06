import numpy as np
import pandas

class song_similarity_model:
    def __init__(self, train_data):
        self.train_data = train_data
    
    def get_user_songs(self, user):
        user_data = self.train_data[self.train_data['user_id'] == user]
        return list(user_data['song'].unique())
    
    def get_song_listeners(self, song):
        song_data = self.train_data[self.train_data['song'] == song]
        return set(song_data['user_id'].unique())
    
    def get_all_songs(self):
        return list(self.train_data['song'].unique())
    
    def build_cooccurence_matrix(self, user_songs, all_songs):
        user_songs_listeners = []
        for song in user_songs:
            user_songs_listeners.append(self.get_song_listeners(song))
            
        matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)
        for i in range(len(all_songs)):
            song_listeners_set_1 = self.get_song_listeners(all_songs[i])
            for j in range(len(user_songs)):
                song_listeners_set_2 = user_songs_listeners[j]
                intersection = song_listeners_set_1.intersection(song_listeners_set_2)
                if len(intersection) != 0:
                    union = song_listeners_set_1.union(song_listeners_set_2)
                    jaccard_index = float(len(intersection))/float(len(union))
                    matrix[j, i] = jaccard_index
                else:
                    matrix[j, i] = 0
        return matrix
    
    def get_top_recommendations(self, user, matrix, all_songs, user_songs):
        # calculate average weight for every user's song
        user_sim_scores = matrix.sum(axis=0)/float(matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()
        
        # sort it by weight
        sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)
        
        df = pandas.DataFrame(columns=['user_id', 'song', 'score', 'rank'])
        rank = 1 
        for i in range(0,len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:
                df.loc[len(df)]=[user,all_songs[sort_index[i][1]],sort_index[i][0],rank]
                rank = rank+1
        
        if df.shape[0] == 0:
            print("The current user has no songs for training the item similarity based recommendation model.")
            return -1
        else:
            return df

        
    def recommend(self, user):
        user_songs = self.get_user_songs(user)
        all_songs = self.get_all_songs()
        matrix = self.build_cooccurence_matrix(user_songs, all_songs)
        return self.get_top_recommendations(user, matrix, all_songs, user_songs)
    
    