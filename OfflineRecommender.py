from flask import Flask
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import csv

app = Flask(__name__)

data_df = pd.read_csv('./Steam.csv', sep=',', names=["User ID", "Name", "Type", "Hours"], engine='python')
# data_df = data_df[data_df["Type"] == "purchase"]


unique_GameID = data_df['Name'].unique()
unique_UserID = data_df['User ID'].unique()

user_count_j = 0
user_old2new_id_dict = dict()
for u in unique_UserID:
    user_old2new_id_dict[u] = user_count_j
    user_count_j += 1

game_count_j = 0
game_old2new_id_dict = dict()
for i in unique_GameID:
    game_old2new_id_dict[i] = game_count_j
    game_count_j += 1

# Then, use the generated dictionaries to reindex UserID and MovieID in the data_df
for j in range(len(data_df)):
    data_df.at[j, 'User ID'] = user_old2new_id_dict[data_df.at[j, 'User ID']]

    data_df.at[j, 'Name'] = game_old2new_id_dict[data_df.at[j, 'Name']]

data_df = data_df[data_df["Type"] == "purchase"]

# import survey data:

with open("GGs 489 survey.csv", newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

data = data[1:]

for i in range(len(data)):
    data[i][1] = data[i][1].split(";")

# Collect the survey data into survey_df
survey_data = []

for i in range(len(data)):
    if data[i][2] in user_old2new_id_dict or data[i][1] == ['']:
        jasdgno = 0
    else:
        user_old2new_id_dict[data[i][2]] = user_count_j
        user_count_j += 1

        for k in data[i][1]:
            survey_data.append([user_old2new_id_dict[data[i][2]], game_old2new_id_dict[k], "purchase", 1.0])

survey_df = pd.DataFrame(survey_data, columns=["User ID", "Name", "Type", "Hours"])

# attach the survey data to the given data
data_df = data_df.append(survey_df, ignore_index=True)
#################################################################################

# generate train_df with 70% samples and test_df with 30% samples, and there should have no overlap between them.
test_df = data_df.sample(frac = 0.3, random_state = np.random.randint(1024))
train_df = data_df.drop(test_df.index)

# generate train_mat and test_mat
num_user = len(data_df['User ID'].unique())
num_game = len(data_df['Name'].unique())

data_mat = coo_matrix((data_df['Hours'].values, (data_df['User ID'].values, data_df['Name'].values)),
                      shape=(num_user, num_game)).toarray().astype(float)
train_mat = coo_matrix((train_df['Hours'].values, (train_df['User ID'].values, train_df['Name'].values)),
                       shape=(num_user, num_game)).toarray().astype(float)
test_mat = coo_matrix((test_df['Hours'].values, (test_df['User ID'].values, test_df['Name'].values)),
                      shape=(num_user, num_game)).toarray().astype(float)
popular_mat = np.sum(data_mat, axis=0)
popular_mat = np.argsort(popular_mat)
train_mat = (train_mat > 0).astype(float)
test_mat = (test_mat > 0).astype(float)

# in order to extract user id and games from matrix use inverse dictionaries:
inv_names = {v: k for k, v in game_old2new_id_dict.items()}
inv_users = {v: k for k, v in user_old2new_id_dict.items()}

usertest = 12394
gametest = 0


class MF_implicit:
    def __init__(self, train_mat, test_mat, latent=5, lr=0.01, reg=0.01):
        self.train_mat = train_mat  # the training rating matrix of size (#user, #movie)
        self.test_mat = test_mat  # the training rating matrix of size (#user, #movie)

        self.latent = latent  # the latent dimension
        self.lr = lr  # learning rate
        self.reg = reg  # regularization weight, i.e., the lambda in the objective function

        self.num_user, self.num_game = train_mat.shape

        self.sample_user, self.sample_game = self.train_mat.nonzero()  # get the user-movie paris w/ratings in train_mat
        self.num_sample = len(self.sample_user)  # the number of user-movie pairs having ratings in train_mat

        self.user_test_like = []
        for u in range(self.num_user):
            self.user_test_like.append(np.where(self.test_mat[u, :] > 0)[0])

        self.P = np.random.random(
            (self.num_user, self.latent))  # latent factors for users, size (#user, self.latent), randomly initialized
        self.Q = np.random.random(
            (self.num_game, self.latent))  # latent factors for users, size (#movie, self.latent), randomly initialized

    def negative_sampling(self):
        negative_movie = np.random.choice(np.arange(self.num_game), size=(len(self.sample_user)), replace=True)
        true_negative = self.train_mat[self.sample_user, negative_movie] == 0
        negative_user = self.sample_user[true_negative]
        negative_movie = negative_movie[true_negative]
        return np.concatenate([self.sample_user, negative_user]), np.concatenate([self.sample_game, negative_movie])

    def train(self, epoch=20):
        for ep in range(epoch):
            userList, gameList = self.negative_sampling()

            # See top for source
            temp = list(zip(userList, gameList))
            np.random.shuffle(temp)
            userList, gameList = zip(*temp)

            # Loops over all user,game pairs
            for i in range(len(userList)):
                user = userList[i]
                game = gameList[i]
                curP = self.P[user]
                curQ = self.Q[game]
                dotted = float(np.dot(curP, curQ.T))
                rating = train_mat[user][game]

                # 𝐏𝑢=𝐏𝑢−𝛾[2(𝐏𝑢⋅𝐐⊤𝑖−𝑟𝑢,𝑖)⋅𝐐𝑖+2𝜆𝐏𝑢]
                newP = curP - self.lr * (2.0 * np.dot(dotted - rating, curQ) + 2.0 * self.reg * curP)

                # 𝐐𝑖=𝐐𝑖−𝛾[2(𝐏𝑢⋅𝐐⊤𝑖−𝑟𝑢,𝑖)⋅𝐏𝑢+2𝜆𝐐𝑖]
                newQ = curQ - self.lr * (2.0 * np.dot(dotted - rating, curP) + 2.0 * self.reg * curQ)

                self.P[user] = newP
                self.Q[game] = newQ

    def predict(self):
        recommendation = np.empty([len(train_mat), 50])

        prediction_mat = np.matmul(self.P, self.Q.T)
        ranked = np.argsort(prediction_mat)

        # creating 50 recommended games for each user -- ignore games that user has rated
        for user in range(0, len(train_mat)):
            curFifty = []
            index = len(ranked[user]) - 1
            cfIndex = len(cfFifty) - 1;
            # populate top 50
            while (len(curFifty) < 50):
                """
                # populate 5 games using implicit feedback
                if useImplicit:
                    impGamesAdded = 0
                    while impGamesAdded < 5:
                        recGame = popular_mat[popularityIdx]
                        popularityIdx -= 1
                        while recGame in curFifty:
                            recGame = ranked[user][popularityIdx]
                            popularityIdx -= 1
                        if train_mat[user][recGame] != 1:
                            curFifty.append(recGame)
                            impGamesAdded += 1
                    useImplicit = False
                # populate 5 games using popular feedback
                else:
                    popGamesAdded = 0
                    while popGamesAdded < 5:
                        recGame = ranked[user][index]
                        index -= 1
                        while recGame in curFifty:
                            recGame = ranked[user][index]
                            index -= 1
                        if train_mat[user][recGame] != 1:
                            curFifty.append(recGame)
                            popGamesAdded += 1
                    useImplicit = True
            recommendation[user] = curFifty
            """
                count = 1
                if count % 2 == 0:
                    recGame = cfFifty[user][cfIndex]
                    cfIndex -= 1
                    while recGame in curFifty:
                        recGame = ranked[user][cfIndex]
                        cfIndex -= 1
                    if train_mat[user][recGame] != 1:
                        curFifty.append(recGame)
                    # populate 5 games using popular feedback
                else:
                    recGame = ranked[user][index]
                    index -= 1
                    while recGame in curFifty:
                        recGame = ranked[user][index]
                        index -= 1
                if train_mat[user][recGame] != 1:
                    curFifty.append(recGame)
                count += 1
            recommendation[user] = curFifty

        return recommendation

    def testRecall(self):
        recommendation = self.predict()

        recalls = np.zeros(3)
        user_count = 0.

        for u in range(self.num_user):
            test_like = self.user_test_like[u]
            test_like_num = len(test_like)
            if test_like_num == 0:
                continue
            rec = recommendation[u, :]
            hits = np.zeros(3)
            for k in range(50):
                if rec[k] in test_like:
                    if k < 50:
                        hits[2] += 1
                        if k < 20:
                            hits[1] += 1
                            if k < 5:
                                hits[0] += 1
            recalls[0] += (hits[0] / test_like_num)
            recalls[1] += (hits[1] / test_like_num)
            recalls[2] += (hits[2] / test_like_num)
            user_count += 1

        recalls /= user_count

        return('recall@5\t[%.6f],\t||\t recall@20\t[%.6f],\t||\t recall@50\t[%.6f]' % (recalls[0], recalls[1], recalls[2]))

    def testPrecision(self):
        recommendation = self.predict()

        precisions = np.zeros(3)
        user_count = 0.

        for u in range(self.num_user):
            test_like = self.user_test_like[u]
            test_like_num = len(test_like)
            if test_like_num == 0:
                continue
            rec = recommendation[u, :]
            hits = np.zeros(3)
            for k in range(50):
                if rec[k] in test_like:
                    if k < 50:
                        hits[2] += 1
                        if k < 20:
                            hits[1] += 1
                            if k < 5:
                                hits[0] += 1
            precisions[0] += (hits[0] / 5.)
            precisions[1] += (hits[1] / 20.)
            precisions[2] += (hits[2] / 50.)
            user_count += 1

        precisions /= user_count

        return('precision@5\t[%.6f],\t||\t precision@20\t[%.6f],\t||\t precision@50\t[%.6f]' % (
                precisions[0], precisions[1], precisions[2]))


mf_implicit = MF_implicit(train_mat, test_mat, latent=5, lr=0.01, reg=0.0001)
mf_implicit.train(epoch=20)

print("here")
user_train_like = []
for u in range(num_user):
    user_train_like.append(np.where(train_mat[u, :] > 0)[0])

numer = np.matmul(train_mat, train_mat.T)
denom = np.sum(train_mat ** 2, axis=1, keepdims=True) ** 0.5
Cosine = numer / np.matmul(denom, denom.T)

cfFifty = []
for u in range(num_user):
    similarities = Cosine[u, :]
    similarities[u] = -1
    N_idx = np.argpartition(similarities, -10)[-10:]
    N_sim = similarities[N_idx]
    scores = np.sum(N_sim.reshape((-1, 1)) * train_mat[N_idx, :], axis=0) / np.sum(N_sim)

    train_like = user_train_like[u]
    scores[train_like] = -9999
    top50_iid = np.argpartition(scores, -50)[-50:]
    top50_iid = top50_iid[np.argsort(scores[top50_iid])[-1::-1]]
    cfFifty.append(top50_iid)
cfFifty = np.array(cfFifty)
print("here")

recGames = mf_implicit.predict()
recall = mf_implicit.testRecall()
print(recall)


with open('recommendations.csv', mode='w', newline='') as recs:
    recs_writer = csv.writer(recs, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    popularRecs = ["Popular"]
    for i in range(len(popular_mat) - 1, len(popular_mat) - 51, -1):
        popularRecs.append(inv_names[popular_mat[i]])
    recs_writer.writerow(popularRecs)

    for user in range(len(train_mat)):
        curGameRecs = []
        curGameRecs.append(inv_users[user])
        for game in recGames[user]:
            curGame = inv_names[game]
            curGameRecs.append(curGame)
        recs_writer.writerow(curGameRecs)

print("here")

@app.route('/')
def MF():
    return recall


if __name__ == '__main__':
    app.run()
