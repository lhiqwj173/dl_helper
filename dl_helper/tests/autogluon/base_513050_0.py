from dl_helper.ag_trainer import autogluon_train_func

def yfunc(y):
    threshold = 0.5
    if y > threshold:
        return 0
    elif y < -threshold:
        return 1
    else:
        return 2

if __name__ == '__main__':
    autogluon_train_func(use_length=3e5, yfunc=yfunc)   