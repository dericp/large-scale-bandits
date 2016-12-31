import numpy as np

alpha = 0.05

article_data = dict()
prev_choice = None
prev_usr_features = None


def set_articles(articles):
    pass


def update(reward):
    if reward != -1:
        (M, b, M_inv, w) = article_data[prev_choice]
        new_M = M + prev_usr_features.dot(prev_usr_features.T)
        new_b = b + reward * prev_usr_features
        #print((reward * prev_usr_features).shape)
        new_M_inv = np.linalg.inv(new_M)
        article_data[prev_choice] = (new_M, new_b, new_M_inv, new_M_inv.dot(new_b))


def recommend(time, user_features, choices):
    curr_ucb = None
    curr_choice = None
    z = np.asarray(user_features).reshape((6, 1))

    for choice in choices:
        if choice not in article_data:
            M = np.identity(6)
            #print(M.shape)
            b = np.zeros((6, 1))
            #print(b.shape)
            M_inv = np.linalg.inv(M)
            #print(M_inv.shape)
            #print(M_inv.dot(b).shape)
            article_data[choice] = (M, b, M_inv, M_inv.dot(b))

        (M, b, M_inv, w) = article_data[choice]
        #print('choice = ' + str(choice))
        #print('ucb = ' + str(w.T.dot(z)) + ' + ' + str(alpha * ((z.T.dot(M_inv).dot(z))**0.5)))
        ucb = (w.T.dot(z) + alpha * ((z.T.dot(M_inv).dot(z))**0.5))[0][0]
        #print('ucb = ' + str(ucb))
        if ucb > curr_ucb:
            #print('found ucb larger than prev')
            curr_ucb = ucb
            curr_choice = choice

    global prev_choice
    prev_choice = curr_choice
    global prev_usr_features
    prev_usr_features = z
    #print('final choice = ' + str(curr_choice))

    return curr_choice

