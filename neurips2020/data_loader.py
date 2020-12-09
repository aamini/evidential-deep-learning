"""
IO module for train/test regression datasets
"""
import numpy as np
import pandas as pd
import os
import h5py
import tensorflow as tf



def generate_cubic(x, noise=False):
    x = x.astype(np.float32)
    y = x**3

    if noise:
        sigma = 3 * np.ones_like(x)
    else:
        sigma = np.zeros_like(x)
    r = np.random.normal(0, sigma).astype(np.float32)
    return y+r, sigma


#####################################
# individual data files             #
#####################################
vb_dir   = os.path.dirname(__file__)
data_dir = os.path.join(vb_dir, "data/uci")

def _load_boston():
    """
    Attribute Information:
    1. CRIM: per capita crime rate by town
    2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
    3. INDUS: proportion of non-retail business acres per town
    4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    5. NOX: nitric oxides concentration (parts per 10 million)
    6. RM: average number of rooms per dwelling
    7. AGE: proportion of owner-occupied units built prior to 1940
    8. DIS: weighted distances to five Boston employment centres
    9. RAD: index of accessibility to radial highways
    10. TAX: full-value property-tax rate per $10,000
    11. PTRATIO: pupil-teacher ratio by town
    12. B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    13. LSTAT: % lower status of the population
    14. MEDV: Median value of owner-occupied homes in $1000's
    """
    data = np.loadtxt(os.path.join(data_dir,
                                   'boston-housing/boston_housing.txt'))
    X    = data[:, :-1]
    y    = data[:,  -1]
    return X, y


def _load_powerplant():
    """
    attribute information:
    features consist of hourly average ambient variables
    - temperature (t) in the range 1.81 c and 37.11 c,
    - ambient pressure (ap) in the range 992.89-1033.30 millibar,
    - relative humidity (rh) in the range 25.56% to 100.16%
    - exhaust vacuum (v) in teh range 25.36-81.56 cm hg
    - net hourly electrical energy output (ep) 420.26-495.76 mw
    the averages are taken from various sensors located around the
    plant that record the ambient variables every second.
    the variables are given without normalization.
    """
    data_file = os.path.join(data_dir, 'power-plant/Folds5x2_pp.xlsx')
    data = pd.read_excel(data_file)
    x    = data.values[:, :-1]
    y    = data.values[:,  -1]
    return x, y


def _load_concrete():
    """
    Summary Statistics:
    Number of instances (observations): 1030
    Number of Attributes: 9
    Attribute breakdown: 8 quantitative input variables, and 1 quantitative output variable
    Missing Attribute Values: None
    Name -- Data Type -- Measurement -- Description
    Cement (component 1) -- quantitative -- kg in a m3 mixture -- Input Variable
    Blast Furnace Slag (component 2) -- quantitative -- kg in a m3 mixture -- Input Variable
    Fly Ash (component 3) -- quantitative -- kg in a m3 mixture -- Input Variable
    Water (component 4) -- quantitative -- kg in a m3 mixture -- Input Variable
    Superplasticizer (component 5) -- quantitative -- kg in a m3 mixture -- Input Variable
    Coarse Aggregate (component 6) -- quantitative -- kg in a m3 mixture -- Input Variable
    Fine Aggregate (component 7) -- quantitative -- kg in a m3 mixture -- Input Variable
    Age -- quantitative -- Day (1~365) -- Input Variable
    Concrete compressive strength -- quantitative -- MPa -- Output Variable
    ---------------------------------
    """
    data_file = os.path.join(data_dir, 'concrete/Concrete_Data.xls')
    data = pd.read_excel(data_file)
    X    = data.values[:, :-1]
    y    = data.values[:,  -1]
    return X, y


def _load_yacht():
    """
    Attribute Information:
    Variations concern hull geometry coefficients and the Froude number:
    1. Longitudinal position of the center of buoyancy, adimensional.
    2. Prismatic coefficient, adimensional.
    3. Length-displacement ratio, adimensional.
    4. Beam-draught ratio, adimensional.
    5. Length-beam ratio, adimensional.
    6. Froude number, adimensional.
    The measured variable is the residuary resistance per unit weight of displacement:
    7. Residuary resistance per unit weight of displacement, adimensional.
    """
    data_file = os.path.join(data_dir, 'yacht/yacht_hydrodynamics.data')
    data = pd.read_csv(data_file, delim_whitespace=True)
    X    = data.values[:, :-1]
    y    = data.values[:,  -1]
    return X, y


def _load_energy_efficiency():
    """
    Data Set Information:
    We perform energy analysis using 12 different building shapes simulated in
    Ecotect. The buildings differ with respect to the glazing area, the
    glazing area distribution, and the orientation, amongst other parameters.
    We simulate various settings as functions of the afore-mentioned
    characteristics to obtain 768 building shapes. The dataset comprises
    768 samples and 8 features, aiming to predict two real valued responses.
    It can also be used as a multi-class classification problem if the
    response is rounded to the nearest integer.
    Attribute Information:
    The dataset contains eight attributes (or features, denoted by X1...X8) and two responses (or outcomes, denoted by y1 and y2). The aim is to use the eight features to predict each of the two responses.
    Specifically:
    X1    Relative Compactness
    X2    Surface Area
    X3    Wall Area
    X4    Roof Area
    X5    Overall Height
    X6    Orientation
    X7    Glazing Area
    X8    Glazing Area Distribution
    y1    Heating Load
    y2    Cooling Load
    """
    data_file = os.path.join(data_dir, 'energy-efficiency/ENB2012_data.xlsx')
    data      = pd.read_excel(data_file)
    X         = data.values[:, :-2]
    y_heating = data.values[:, -2]
    y_cooling = data.values[:, -1]
    return X, y_cooling


def _load_wine():
    """
    Attribute Information:
    For more information, read [Cortez et al., 2009].
    Input variables (based on physicochemical tests):
    1 - fixed acidity
    2 - volatile acidity
    3 - citric acid
    4 - residual sugar
    5 - chlorides
    6 - free sulfur dioxide
    7 - total sulfur dioxide
    8 - density
    9 - pH
    10 - sulphates
    11 - alcohol
    Output variable (based on sensory data):
    12 - quality (score between 0 and 10)
    """
    # data_file = os.path.join(data_dir, 'wine-quality/winequality-red.csv')
    data_file = os.path.join(data_dir, 'wine-quality/wine_data_new.txt')
    data     = pd.read_csv(data_file, sep=' ', header=None)
    X = data.values[:, :-1]
    y = data.values[:,  -1]
    return X, y

def _load_kin8nm():
    """
    This is data set is concerned with the forward kinematics of an 8 link robot arm. Among the existing variants of
     this data set we have used the variant 8nm, which is known to be highly non-linear and medium noisy.

    Original source: DELVE repository of data. Source: collection of regression datasets by Luis Torgo
    (ltorgo@ncc.up.pt) at http://www.ncc.up.pt/~ltorgo/Regression/DataSets.html Characteristics: 8192 cases,
    9 attributes (0 nominal, 9 continuous).

    Input variables:
    1 - theta1
    2 - theta2
    ...
    8 - theta8
    Output variable:
    9 - target
    """
    data_file = os.path.join(data_dir, 'kin8nm/dataset_2175_kin8nm.csv')
    data     = pd.read_csv(data_file, sep=',')
    X = data.values[:, :-1]
    y = data.values[:,  -1]
    return X, y


def _load_naval():
    """
    http://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants

    Input variables:
    1 - Lever position(lp)[]
    2 - Ship speed(v)[knots]
    3 - Gas Turbine shaft torque(GTT)[kNm]
    4 - Gas Turbine rate of revolutions(GTn)[rpm]
    5 - Gas Generator rate of revolutions(GGn)[rpm]
    6 - Starboard Propeller Torque(Ts)[kN]
    7 - Port Propeller Torque(Tp)[kN]
    8 - HP Turbine exit temperature(T48)[C]
    9 - GT Compressor inlet air temperature(T1)[C]
    10 - GT Compressor outlet air temperature(T2)[C]
    11 - HP Turbine exit pressure(P48)[bar]
    12 - GT Compressor inlet air pressure(P1)[bar]
    13 - GT Compressor outlet air pressure(P2)[bar]
    14 - Gas Turbine exhaust gas pressure(Pexh)[bar]
    15 - Turbine Injecton Control(TIC)[ %]
    16 - Fuel flow(mf)[kg / s]
    Output variables:
    17 - GT Compressor decay state coefficient.
    18 - GT Turbine decay state coefficient.
    """
    data = np.loadtxt(os.path.join(data_dir, 'naval/data.txt'))
    X = data[:, :-2]
    y_compressor = data[:, -2]
    y_turbine = data[:, -1]
    return X, y_turbine

def _load_protein():
    """
    Physicochemical Properties of Protein Tertiary Structure Data Set
    Abstract: This is a data set of Physicochemical Properties of Protein Tertiary Structure.
    The data set is taken from CASP 5-9. There are 45730 decoys and size varying from 0 to 21 armstrong.

    TODO: Check that the output is correct

    Input variables:
        RMSD-Size of the residue.
        F1 - Total surface area.
        F2 - Non polar exposed area.
        F3 - Fractional area of exposed non polar residue.
        F4 - Fractional area of exposed non polar part of residue.
        F5 - Molecular mass weighted exposed area.
        F6 - Average deviation from standard exposed area of residue.
        F7 - Euclidian distance.
        F8 - Secondary structure penalty.
    Output variable:
        F9 - Spacial Distribution constraints (N,K Value).
    """
    data_file = os.path.join(data_dir, 'protein/CASP.csv')
    data     = pd.read_csv(data_file, sep=',')
    X = data.values[:, 1:]
    y = data.values[:, 0]
    return X, y

def _load_song():
    """
    INSTRUCTIONS:
    1) Download from http://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
    2) Place YearPredictionMSD.txt in data/uci/song/

    Dataloader is slow since file is large.

    YearPredictionMSD Data Set
    Prediction of the release year of a song from audio features. Songs are mostly western, commercial tracks ranging
    from 1922 to 2011, with a peak in the year 2000s.

    90 attributes, 12 = timbre average, 78 = timbre covariance
    The first value is the year (target), ranging from 1922 to 2011.
    Features extracted from the 'timbre' features from The Echo Nest API.
    We take the average and covariance over all 'segments', each segment
    being described by a 12-dimensional timbre vector.

    """
    data = np.loadtxt(os.path.join(data_dir,
                                   'song/YearPredictionMSD.txt'), delimiter=',')
    X    = data[:, :-1]
    y    = data[:,  -1]
    return X, y


def _load_depth():
    train = h5py.File("data/depth_train.h5", "r")
    test = h5py.File("data/depth_test.h5", "r")
    return (train["image"], train["depth"]), (test["image"], test["depth"])

def load_depth():
    return _load_depth()

def load_apollo():
    test = h5py.File("data/apolloscape_test.h5", "r")
    return (None, None), (test["image"], test["depth"])

def load_dataset(name, split_seed=0, test_fraction=.1, return_as_tensor=False):
    # load full dataset
    load_funs = { "wine"              : _load_wine,
                  "boston"            : _load_boston,
                  "concrete"          : _load_concrete,
                  "power-plant"       : _load_powerplant,
                  "yacht"             : _load_yacht,
                  "energy-efficiency" : _load_energy_efficiency,
                  "kin8nm"            : _load_kin8nm,
                  "naval"             : _load_naval,
                  "protein"           : _load_protein,
                  "depth"              : _load_depth,
                  "song"              : _load_song}

    print("Loading dataset {}....".format(name))
    if name == "depth":
        (X_train, y_train), (X_test, y_test) = load_funs[name]()
        y_scale = np.array([[1.0]])
        return (X_train, y_train), (X_test, y_test), y_scale

    X, y = load_funs[name]()
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    def standardize(data):
        mu = data.mean(axis=0, keepdims=1)
        scale = data.std(axis=0, keepdims=1)
        scale[scale<1e-10] = 1.0

        data = (data - mu) / scale
        return data, mu, scale



    # We create the train and test sets with 90% and 10% of the data

    if split_seed == -1:  # Do not shuffle!
        permutation = range(X.shape[0])
    else:
        rs = np.random.RandomState(split_seed)
        permutation = rs.permutation(X.shape[0])

    if name == "boston" or name == "wine":
        test_fraction = 0.2
    size_train  = int(np.round(X.shape[ 0 ] * (1 - test_fraction)))
    index_train = permutation[ 0 : size_train ]
    index_test  = permutation[ size_train : ]

    X_train = X[ index_train, : ]
    X_test  = X[ index_test, : ]

    if name == "depth":
        y_train = y[index_train]
        y_test = y[index_test]
    else:
        y_train = y[index_train, None]
        y_test = y[index_test, None]


    X_train, x_train_mu, x_train_scale = standardize(X_train)
    X_test = (X_test - x_train_mu) / x_train_scale

    y_train, y_train_mu, y_train_scale = standardize(y_train)
    y_test = (y_test - y_train_mu) / y_train_scale

    if return_as_tensor:
        X_train = tf.convert_to_tensor(X_train, tf.float32)
        X_test = tf.convert_to_tensor(X_test, tf.float32)
        y_train = tf.convert_to_tensor(y_train, tf.float32)
        y_test = tf.convert_to_tensor(y_test, tf.float32)

    print("Done loading dataset {}".format(name))
    return (X_train, y_train), (X_test, y_test), y_train_scale




def load_flight_delay():

    # Download from here: http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/dataset_mirror/airline_delay/
    data = pd.read_pickle("data/flight-delay/filtered_data.pickle")
    y = np.array(data['ArrDelay'])
    data.pop('ArrDelay')
    X = np.array(data[:])

    def standardize(data):
        data -= data.mean(axis=0, keepdims=1)
        scale = data.std(axis=0, keepdims=1)
        data /= scale
        return data, scale

    X = X[:, np.where(data.var(axis=0) > 0)[0]]
    X, _ = standardize(X)
    y, y_scale = standardize(y.reshape(-1,1))
    y = np.squeeze(y)
    # y_scale = np.array([[1.0]])

    N = 700000
    S = 100000
    X_train = X[:N,:]
    X_test = X[N:N + S, :]
    y_train = y[:N]
    y_test = y[N:N + S]


    return (X_train, y_train), (X_test, y_test), y_scale


# (X_train, y_train), (X_test, y_test) = load_dataset('boston')
# (X_train, y_train), (X_test, y_test) = load_dataset('concrete')
# (X_train, y_train), (X_test, y_test) = load_dataset('energy-efficiency')
# (X_train, y_train), (X_test, y_test) = load_dataset('kin8nm')
# (X_train, y_train), (X_test, y_test) = load_dataset('naval')
# (X_train, y_train), (X_test, y_test) = load_dataset('power-plant')
# (X_train, y_train), (X_test, y_test) = load_dataset('protein')
# (X_train, y_train), (X_test, y_test) = load_dataset('wine')
# (X_train, y_train), (X_test, y_test) = load_dataset('yacht', split_seed=-1)
# (X_train, y_train), (X_test, y_test) = load_dataset('song', split_seed=-1)
# (X_train, y_train), (X_test, y_test) = load_dataset('depth')

# import pdb; pdb.set_trace()
