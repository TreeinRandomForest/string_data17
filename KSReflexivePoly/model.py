import numpy as np
import copy
from keras.models import Sequential
from keras.layers import Dense

def clean_d3_data(filename, zeropad = False):
    '''
    '''
    with open(filename) as f:
        features = []
        target = []

        n_examples = 0
        n_max = 0
        for line in f:
            if line.find(":")>-1:
                line_split = line.rstrip("\n").split()

                if n_examples > 0:
                    features.append(np.array(matrix))
                    target.append(pic)
                
                d = int(line_split[0])
                n = int(line_split[1])
                pic = int(line_split[-2].split(":")[1])

                if n_max < n:
                    n_max = n
                
                if d != 3:
                    raise ValueError()
                
                n_examples += 1
                
                matrix = [] #store 3xN matrix of polytope lattice points
            else:
                coordinates = [int(val) for val in line.split()]

                if len(coordinates) > 0:
                    matrix.append(np.array(coordinates))

        #process last example
        features.append(np.array(matrix))
        target.append(pic)

        target = np.array(target)

        #zero-pad
        if zeropad:
            features = [np.hstack([f, np.zeros((3, n_max - f.shape[1]))]) for f in features]
        features = np.array(features)
        
        return(features, target)

def augment_data(features, target, seed=0, n_transforms=5):
    np.random.seed(seed)

    augmented_features = []
    augmented_target = []
    
    for f_index in range(len(features)):
        f = features[f_index]
        t = target[f_index]
        
        for n in range(n_transforms): #generate permutations
            f_augment = f.transpose()[np.random.permutation(f.shape[1]),:].transpose() #permute the columns i.e. the lattice points

            augmented_features.append(f_augment)
            augmented_target.append(t)
            
    augmented_features, augmented_target = np.array(augmented_features), np.array(augmented_target)

    features = np.append(features, augmented_features, axis=0)
    target = np.append(target, augmented_target, axis=0)

    return(features, target)

def flatten_data(features):
    flattened_features = []
    for f in features:
        flattened_features.append(np.reshape(f, f.shape[0]*f.shape[1]))
    flattened_features = np.array(flattened_features)

    return(flattened_features)
        
def build_model(input_dim):
    model = Sequential()
    model.add(Dense(units = 1000, activation='sigmoid', input_dim=input_dim))
    model.add(Dense(units = 100, activation='tanh'))
    model.add(Dense(units=1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    return(model)

def train_test_split(features, target, train_size=0.5):
    N = len(features)

    indices = list(range(N))
    np.random.shuffle(indices)

    N_train = int(train_size * N)

    train_indices, test_indices = indices[0:N_train], indices[N_train:]

    train_features, test_features = features[train_indices], features[test_indices]
    train_target, test_target = target[train_indices], target[test_indices]
    
    return({'train_features': train_features,
            'train_target': train_target,
            'test_features': test_features,
            'test_target': test_target
    })

def binarize_target(target, threshold):
    target = (target > threshold).astype(int)

    return target

def train_model(model, train_features, train_target):
    model.fit(train_features, train_target, epochs = 100, batch_size=64)

    return(model)

if __name__ == "__main__":
    threshold = 10
    filename = "data/d3/RefPoly.d3"
    augment = False

    features, target = clean_d3_data(filename, zeropad = True)

    original_features = copy.copy(features)
    original_target = copy.copy(target)
    
    if augment:
        features, target = augment_data(features, target, seed=0, n_transforms=5)

    original_features = flatten_data(original_features)
    original_target = binarize_target(original_target, threshold)
        
    features = flatten_data(features)
    target = binarize_target(target, threshold)

    d = train_test_split(features, target, train_size=0.5)

    model = build_model(len(features[0]))
    model = train_model(model, d['train_features'], d['train_target'])

    
