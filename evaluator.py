import random

class precision_recall_calculator():
    
    def __init__(self, test_data, train_data, is_model):
        self.test_data = test_data
        self.train_data = train_data
        self.user_test_sample = None
        self.model = is_model
        
        self.ism_training_dict = dict()
        self.test_dict = dict()
    
    def remove_percentage(self, list_a, percentage):
        k = int(len(list_a) * percentage)
        random.seed(0)
        indicies = random.sample(range(len(list_a)), k)
        new_list = [list_a[i] for i in indicies]
    
        return new_list
    
    def create_user_test_sample(self, percentage):
        users_test_and_training = list(set(self.test_data['user_id'].unique()).intersection(set(self.train_data['user_id'].unique())))
        print("Length of user_test_and_training:%d" % len(users_test_and_training))

        self.users_test_sample = self.remove_percentage(users_test_and_training, percentage)

        print("Length of user sample:%d" % len(self.users_test_sample))
        
    def get_test_sample_recommendations(self):
        for user_id in self.users_test_sample:
            print("Getting recommendations for user:%s" % user_id)
            user_sim_items = self.model.recommend(user_id)
            self.ism_training_dict[user_id] = list(user_sim_items["song"])
    
            test_data_user = self.test_data[self.test_data['user_id'] == user_id]
            self.test_dict[user_id] = set(test_data_user['song'].unique() )
    
    def calculate_precision_recall(self):
        cutoff_list = list(range(1,11))

        ism_avg_precision_list = []
        ism_avg_recall_list = []

        num_users_sample = len(self.users_test_sample)
        for N in cutoff_list:
            ism_sum_precision = 0
            ism_sum_recall = 0
            ism_avg_precision = 0
            ism_avg_recall = 0

            for user_id in self.users_test_sample:
                ism_hitset = self.test_dict[user_id].intersection(set(self.ism_training_dict[user_id][0:N]))
                testset = self.test_dict[user_id]
        
                ism_sum_precision += float(len(ism_hitset))/float(len(testset))
                ism_sum_recall += float(len(ism_hitset))/float(N)
        
            ism_avg_precision = ism_sum_precision/float(num_users_sample)
            ism_avg_recall = ism_sum_recall/float(num_users_sample)

            ism_avg_precision_list.append(ism_avg_precision)
            ism_avg_recall_list.append(ism_avg_recall)
    
        return (ism_avg_precision_list, ism_avg_recall_list)
     

    def calculate_measures(self, percentage):
        self.create_user_test_sample(percentage)
        
        self.get_test_sample_recommendations()
        
        return self.calculate_precision_recall()