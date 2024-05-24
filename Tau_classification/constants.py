
"""
Module with useful constants
"""
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE

from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

extracted_features = [
                    'Image',
                    'Name',
                    'Class',
                    'Parent',
                    'ROI',
                    'Centroid X µm',
                    'Centroid Y µm',
                    'Area µm^2',
                    'Circularity', 'Length µm',
                    'Max diameter µm',
                    'Min diameter µm', 'ROI: 0.25 µm per pixel: Blue: Max',
                    'ROI: 0.25 µm per pixel: Blue: Mean',
                    'ROI: 0.25 µm per pixel: Blue: Median',
                    'ROI: 0.25 µm per pixel: Blue: Min',
                    'ROI: 0.25 µm per pixel: Blue: Std.dev.',
                    'ROI: 0.25 µm per pixel: Brightness: Max',
                    'ROI: 0.25 µm per pixel: Brightness: Mean',
                    'ROI: 0.25 µm per pixel: Brightness: Median',
                    'ROI: 0.25 µm per pixel: Brightness: Min',
                    'ROI: 0.25 µm per pixel: Brightness: Std.dev.',
                    'ROI: 0.25 µm per pixel: DAB: Haralick Angular second moment (F0)',
                    'ROI: 0.25 µm per pixel: DAB: Haralick Contrast (F1)',
                    'ROI: 0.25 µm per pixel: DAB: Haralick Correlation (F2)',
                    'ROI: 0.25 µm per pixel: DAB: Haralick Difference entropy (F10)',
                    'ROI: 0.25 µm per pixel: DAB: Haralick Difference variance (F9)',
                    'ROI: 0.25 µm per pixel: DAB: Haralick Entropy (F8)',
                    'ROI: 0.25 µm per pixel: DAB: Haralick Information measure of correlation 1 (F11)',
                    'ROI: 0.25 µm per pixel: DAB: Haralick Information measure of correlation 2 (F12)',
                    'ROI: 0.25 µm per pixel: DAB: Haralick Inverse difference moment (F4)',
                    'ROI: 0.25 µm per pixel: DAB: Haralick Sum average (F5)',
                    'ROI: 0.25 µm per pixel: DAB: Haralick Sum entropy (F7)',
                    'ROI: 0.25 µm per pixel: DAB: Haralick Sum of squares (F3)',
                    'ROI: 0.25 µm per pixel: DAB: Haralick Sum variance (F6)',
                    'ROI: 0.25 µm per pixel: DAB: Max', 'ROI: 0.25 µm per pixel: DAB: Mean',
                    'ROI: 0.25 µm per pixel: DAB: Median',
                    'ROI: 0.25 µm per pixel: DAB: Min',
                    'ROI: 0.25 µm per pixel: DAB: Std.dev.',
                    'ROI: 0.25 µm per pixel: Green: Max',
                    'ROI: 0.25 µm per pixel: Green: Mean',
                    'ROI: 0.25 µm per pixel: Green: Median',
                    'ROI: 0.25 µm per pixel: Green: Min',
                    'ROI: 0.25 µm per pixel: Green: Std.dev.',
                    'ROI: 0.25 µm per pixel: Hematoxylin: Max',
                    'ROI: 0.25 µm per pixel: Hematoxylin: Mean',
                    'ROI: 0.25 µm per pixel: Hematoxylin: Median',
                    'ROI: 0.25 µm per pixel: Hematoxylin: Min',
                    'ROI: 0.25 µm per pixel: Hematoxylin: Std.dev.',
                    'ROI: 0.25 µm per pixel: Red: Max', 'ROI: 0.25 µm per pixel: Red: Mean',
                    'ROI: 0.25 µm per pixel: Red: Median',
                    'ROI: 0.25 µm per pixel: Red: Min',
                    'ROI: 0.25 µm per pixel: Red: Std.dev.',
                    'ROI: 0.25 µm per pixel: Saturation: Max',
                    'ROI: 0.25 µm per pixel: Saturation: Mean',
                    'ROI: 0.25 µm per pixel: Saturation: Median',
                    'ROI: 0.25 µm per pixel: Saturation: Min',
                    'ROI: 0.25 µm per pixel: Saturation: Std.dev.',
                    'Solidity']

# Screening classifier (general purpose)
# screening_classifier_hyperparams = [
#     ('normalizer', MinMaxScaler()),
#     ('selector', RFE(RandomForestClassifier(
#         random_state=42),
#         n_features_to_select=34)),
#     ('clf', BalancedRandomForestClassifier(
#         random_state=42,
#         sampling_strategy='majority',
#         n_estimators=200,
#         min_samples_split=10,
#         min_samples_leaf=1,
#         max_features=1,
#         max_depth=None,
#         max_samples=0.75,
#         class_weight='balanced'))]

# Screening classifier UPDATED (general purpose)
screening_classifier_hyperparams = [
    ('normalizer', MinMaxScaler()),
    ('selector', RFE(RandomForestClassifier(
        random_state=42),
        n_features_to_select=46)),
    ('clf', BalancedRandomForestClassifier(
        random_state=42,
        sampling_strategy='auto',
        n_estimators=600,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features=1,
        max_depth=None,
        max_samples=None,
        class_weight='balanced'))]


# Tau classifier for cortical regions

cortical_classifier_hyperparams = [
    ('normalizer', MinMaxScaler()),
    ('selector', RFE(RandomForestClassifier(
        random_state=42),
        n_features_to_select=40)),
    ('clf', BalancedRandomForestClassifier(
        random_state=42,
        sampling_strategy='not majority',
        n_estimators=800,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.2,
        max_depth=10,
        max_samples=0.75,
        class_weight='balanced'))]

# Tau classifier for subthalamic nucleus & globus pallidus
stn_gp_classifier_hyperparams = [
    ('normalizer', MinMaxScaler()),
    ('selector', RFE(RandomForestClassifier(
        random_state=42),
        n_features_to_select=34)),
    ('clf', BalancedRandomForestClassifier(
        random_state=42,
        sampling_strategy='not majority',
        n_estimators=500,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features=0.6,
        max_depth=15,
        max_samples=0.75,
        class_weight='balanced'))]

# Tau classifier for striatum
str_classifier_hyperparams = [
    ('normalizer', MinMaxScaler()),
    ('selector', RFE(RandomForestClassifier(
        random_state=42),
        n_features_to_select=34)),
    ('clf', BalancedRandomForestClassifier(
        random_state=42,
        sampling_strategy='not majority',
        n_estimators=500,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features=0.6,
        max_depth=15,
        max_samples=0.75,
        class_weight='balanced'))]

# Tau classifier for dentate nucleus
dn_classifier_hyperparams = [
    ('normalizer', MinMaxScaler()),
    ('selector', RFE(RandomForestClassifier(
        random_state=42),
        n_features_to_select=34)),
    ('clf', BalancedRandomForestClassifier(
        random_state=42,
        sampling_strategy='not majority',
        n_estimators=100,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.2,
        max_depth=None,
        max_samples=None,
        class_weight='balanced'))]
