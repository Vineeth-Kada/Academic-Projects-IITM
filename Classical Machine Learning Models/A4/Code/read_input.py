# Define Normalise
def normalise(X_train, X_dev):
    x = np.array(X_train)
    mn = x.min(0)
    min_max_diff = x.ptp(0)
    for i in range(len(X_train)):
        X_train[i] = (X_train[i] - mn) / min_max_diff
    for i in range(len(X_dev)):
        X_dev[i] = (X_dev[i] - mn) / min_max_diff
    return X_train, X_dev

# Helper function to convert timeseries to a set of fixed length features
def rolling_average(all_X,l):
    ans = []
    for fvec_old in all_X:
        n = len(fvec_old)
        fvec_new = []
        w = n - l + 1
        for i in range(l):
            fvec_new.append(np.average(fvec_old[i:i+w],axis = 0))

        ans.append(np.array(fvec_new).flatten())
    return ans

# Synthetic Data
def Synthetic():
    X_train, T_train = [], []
    X_dev, T_dev = [], []
    nClasses = 0

    with open('Synthetic/train.txt') as train_syn:
        lines = train_syn.readlines()

        for line in lines:
            x, y, c = line.strip().split(',')
            nClasses = max(nClasses, int(c))
            x, y, c = float(x), float(y), int(c) - 1
            X_train.append(np.array([x, y]))
            T_train.append(c)


    with open('Synthetic/dev.txt') as dev_syn:
        lines = dev_syn.readlines()

        for line in lines:
            x, y, c = line.strip().split(',')
            x, y, c = float(x), float(y), int(c) - 1
            X_dev.append(np.array([x, y]))
            T_dev.append(c)

    return X_train,T_train,X_dev,T_dev,nClasses

# RealData
def RealData():
    X_train, T_train = [], []
    X_dev, T_dev = [], []
    nClasses = 0

    class_names = ['coast','forest','highway','mountain','opencountry']
    nClasses = len(class_names)
    for c in range(nClasses):
        name = class_names[c]
        path_train = f'RealData/{name}/train/'
        path_dev = f'RealData/{name}/dev/'

        """ Collecting Training Data """
        files = os.listdir(path_train)

        for file in files:
            fvec = []
            with open(path_train + file, 'r') as f:
                for line in f.readlines():
                    fvec += list(map(float,line.split()))
            X_train.append(np.array(fvec))
            T_train.append(c)

        """ Collecting Development Data """
        files = os.listdir(path_dev)
        images = []
        for file in files:
            fvec = []
            with open(path_dev + file, 'r') as f:
                for line in f.readlines():
                    fvec += list(map(float,line.split()))
            X_dev.append(np.array(fvec))
            T_dev.append(c)

    return X_train,T_train,X_dev,T_dev,nClasses
        



# speech_data 
def speech_data():
    X_train, T_train = [], []
    X_dev, T_dev = [], []
    nClasses = 0

    class_names = ['1','4','5','7','o']
    nClasses = len(class_names)
    
    l = 10000000
    for c in range(nClasses):
        name = class_names[c]

        path = f'speech_data/{name}/train/*.mfcc'
        all_filePaths = list(glob.glob(path))


        for file in all_filePaths:
            with open(file) as f:
                lines = f.readlines()
                all_fvec = []
                l = min(l, int(list(lines[0].split())[1] ) )


                for line in lines[1:]:
                    all_fvec.append(list(map(float,line.split())))
                X_train.append(all_fvec)
                T_train.append(c)


        path = f'speech_data/{name}/dev/*.mfcc'
        all_filePaths = list(glob.glob(path))


        for file in all_filePaths:
            with open(file) as f:
                lines = f.readlines()
                all_fvec = []
                l = min(l, int(list(lines[0].split())[1] ) )


                for line in lines[1:]:
                    all_fvec.append(list(map(float,line.split())))
                X_dev.append(all_fvec)
                T_dev.append(c)

    X_train = rolling_average(X_train,l)
    X_dev = rolling_average(X_dev,l)
    return X_train,T_train,X_dev,T_dev,nClasses


def extractHandWritingData(dir):
    files = os.listdir(dir)
    
    templates = []
    for file in files:
        with open(dir + file, 'r') as f:
            curr = np.array(f.readline().split()[1:], dtype=np.float64).reshape(-1,2)
            
            # Position Invariant
            for i in range(2):
                mn = np.min(curr[:, i])
                mx = np.max(curr[:, i])
                curr[:, i] = curr[:, i] - (mn + mx) / 2
            
            # Scale Invariant
            mn = np.array([np.min(curr[:, 0]), np.min(curr[:, 1])])
            mx = np.array([np.max(curr[:, 0]), np.max(curr[:, 1])])
            diff = mx - mn
            curr[:, 0] = curr[:, 0] / diff[0]
            curr[:, 1] = curr[:, 1] / diff[1]
            
            lst = []
            for y in curr:
                lst.append(y)
            templates.append(lst)
    return templates

# character_data
def character_data():
    X_train, T_train = [], []
    X_dev, T_dev = [], []
    nClasses = 0

    class_names = ['ai','bA','dA','lA','tA']
    nClasses = len(class_names)
    
    l = 10000000
    for c in range(nClasses):
        name = class_names[c]

        path = f'character_data/{name}/train/*.txt'
        all_filePaths = list(glob.glob(path))

        X_train_subset = extractHandWritingData(f'character_data/{name}/train/')
        n = len(X_train_subset)
        X_train += ( X_train_subset )
        for i in range(n):
            l = min(l, len(X_train_subset[i]))
            T_train.append(c)

        path = f'speech_data/{name}/dev/*.txt'
        all_filePaths = list(glob.glob(path))

        X_dev_subset = extractHandWritingData(f'character_data/{name}/dev/')
        n = len(X_dev_subset)
        X_dev += ( X_dev_subset )
        for i in range(n):
            l = min(l, len(X_dev_subset[i]))
            T_dev.append(c)

    X_train = rolling_average(X_train,l)
    X_dev = rolling_average(X_dev,l)
    
    return X_train,T_train,X_dev,T_dev,nClasses
