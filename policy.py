import numpy as np

alpha = 1

article_data = dict()
prev_choice = None
prev_usr_features = None


def set_articles(articles):
    pass


def update(reward):
    if reward != -1:
        (M, b, M_inv, w) = article_data[prev_choice]
        new_M = M + prev_usr_features.dot(prev_usr_features)
        new_b = b + reward * prev_usr_features
        new_M_inv = np.linalg.inv(new_M)
        article_data[prev_choice] = (new_M, new_b, new_M_inv, new_M_inv.dot(new_b))


def recommend(time, user_features, choices):
    curr_ucb = None
    curr_choice = None
    z = np.asarray(user_features)

    for choice in choices:
        if choice not in article_data:
            M = np.identity(6)
            b = np.zeros(6)
            M_inv = np.linalg.inv(M)
            article_data[choice] = (M, b, M_inv, M_inv.dot(b))

        (M, b, M_inv, w) = article_data[choice]
        #print('choice = ' + str(choice))
        #print('ucb = ' + str(w.dot(z)) + ' + ' + str(alpha * ((z.dot(M_inv).dot(z))**0.5)))
        ucb = w.dot(z) + alpha * ((z.dot(M_inv).dot(z))**0.5)
        if ucb > curr_ucb:
            print('found ucb larger than prev')
            curr_ucb = ucb
            curr_choice = choice

    global prev_choice
    prev_choice = curr_choice
    global prev_usr_features
    prev_usr_features = z
    print('final choice = ' + str(choice))

    return curr_choice

