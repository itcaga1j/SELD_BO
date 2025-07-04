# Extracts the features, labels, and normalizes the development and evaluation split features.
import sys
import cls_feature_class
import parameters
import sys


def main(argv):
    # using default parameters
    params = parameters.get_params()

    # -------------- Extract features and labels for development set -----------------------------
    if params['mode'] == 'dev':
        dev_feat_cls = cls_feature_class.FeatureClass(params, is_eval=False)

        # # Extract features and normalize them
        dev_feat_cls.extract_all_feature()
        dev_feat_cls.preprocess_features()

        # # Extract labels
        dev_feat_cls.extract_all_labels()


    else:
        dev_feat_cls = cls_feature_class.FeatureClass(params, is_eval=True)

        # # Extract features and normalize them
        dev_feat_cls.extract_all_feature()
        dev_feat_cls.preprocess_features()


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)


