# Shapelet Feature Integration
# ================================================================

import os

import numpy as np
import pandas as pd

# Configuration
base_dir = os.getcwd()
DATA_DIR = f"{base_dir}/../data/mimiciv/processed"
SHAPELET_RESULTS_DIR = f"{base_dir}/../results/csv_results"
OUTPUT_DIR = DATA_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load processed data
train_df = pd.read_csv(os.path.join(DATA_DIR, "processed_train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "processed_test.csv"))

# Separate features and labels
X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]
X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

# Process shapelet distances
listdir = os.listdir(SHAPELET_RESULTS_DIR)

Train_shapelet = pd.DataFrame(index=range(len(X_train)))
Test_shapelet = pd.DataFrame(index=range(len(X_test)))

# Process each shapelet file
for dist_file in listdir:
    if "Train_" in dist_file:
        train_dist_path = os.path.join(SHAPELET_RESULTS_DIR, dist_file)

        parts = dist_file.split("_")
        dist_name = parts[1].split(".")[0]

        test_dist_path = train_dist_path.replace("Train", "Test")

        if not os.path.exists(test_dist_path):
            continue

        try:
            df_train = pd.read_csv(train_dist_path)
            df_test = pd.read_csv(test_dist_path)

            df_train = df_train.sort_values("p_val")
            df_test = df_test.reindex(df_train.index)

            df_train = df_train.reset_index(drop=True)
            df_test = df_test.reset_index(drop=True)

            for df_idx, (df, save_df) in enumerate(
                zip([df_train, df_test], [Train_shapelet, Test_shapelet])
            ):
                shapelet_num = min(len(df), 20)
                new_columns = {}

                for i in range(shapelet_num):
                    if "distances" not in df.columns or "shapelet" not in df.columns:
                        break

                    shapelet_str = df["shapelet"][i]
                    try:
                        shapelet_str = shapelet_str.strip()
                        if shapelet_str.startswith("[") and shapelet_str.endswith("]"):
                            nums = shapelet_str[1:-1].split(", ")
                            nums = [float(num) for num in nums]
                            string = "-".join(map(str, nums))
                        else:
                            string = shapelet_str
                    except Exception as e:
                        continue

                    distances_str = df["distances"][i]
                    try:
                        distances_str = distances_str.strip()
                        if distances_str.startswith("[") and distances_str.endswith(
                            "]"
                        ):
                            list_ = distances_str[1:-1].split(",")
                            dist = []
                            for a in list_:
                                a = a.strip()
                                if a.lower() == "nan" or a == "":
                                    dist.append(np.nan)
                                else:
                                    try:
                                        dist.append(float(a))
                                    except ValueError:
                                        dist.append(np.nan)
                        else:
                            try:
                                dist = [float(distances_str)]
                            except ValueError:
                                dist = [np.nan]
                    except Exception as e:
                        continue

                    if len(dist) > len(save_df):
                        dist = dist[: len(save_df)]
                    elif len(dist) < len(save_df):
                        dist.extend([np.nan] * (len(save_df) - len(dist)))

                    new_columns[f"{dist_name}_{string}"] = dist

                if new_columns:
                    new_columns_df = pd.DataFrame(new_columns, index=save_df.index)

                    if df_idx == 0:
                        Train_shapelet = pd.concat(
                            [Train_shapelet, new_columns_df], axis=1
                        )
                    else:
                        Test_shapelet = pd.concat(
                            [Test_shapelet, new_columns_df], axis=1
                        )

        except Exception as e:
            continue


# Remove highly correlated features
def remove_highly_correlated_features(
    X_train, threshold=0.7, min_features_per_category=3
):
    """Remove highly correlated features while maintaining minimum per category."""

    if X_train.shape[1] == 0:
        return []

    dist_dict = {}
    for feature in X_train.columns:
        try:
            dist_name, _ = feature.split("_", 1)
            if dist_name not in dist_dict:
                dist_dict[dist_name] = [feature]
            else:
                dist_dict[dist_name].append(feature)
        except Exception as e:
            continue

    keep_features = []

    for key, features in dist_dict.items():
        try:
            if len(features) <= min_features_per_category:
                keep_features.extend(features)
                continue

            data_subset = X_train[features]
            data_subset_filled = data_subset.fillna(data_subset.mean())

            try:
                corr_matrix = data_subset_filled.corr().abs()
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )

                features_to_drop = []
                for column in upper_triangle.columns:
                    if any(upper_triangle[column] > threshold):
                        features_to_drop.append(column)

                remaining_features = [f for f in features if f not in features_to_drop]
                remaining_count = len(remaining_features)

                if remaining_count < min_features_per_category:
                    feature_importance = data_subset_filled.var().sort_values(
                        ascending=False
                    )
                    features_to_keep = feature_importance.head(
                        min_features_per_category
                    ).index.tolist()
                    keep_features.extend(features_to_keep)
                else:
                    keep_features.extend(remaining_features)

            except Exception as e:
                keep_features.extend(features)

        except Exception as e:
            keep_features.extend(features)

    return keep_features


# Apply feature selection
MIN_FEATURES_PER_CATEGORY = 1

if Train_shapelet.shape[1] > 0:
    keep_feat = remove_highly_correlated_features(
        Train_shapelet,
        threshold=0.9,
        min_features_per_category=MIN_FEATURES_PER_CATEGORY,
    )
    if keep_feat:
        Train_shapelet_filtered = Train_shapelet[keep_feat]
        common_cols = list(
            set(Train_shapelet_filtered.columns) & set(Test_shapelet.columns)
        )
        Test_shapelet_filtered = Test_shapelet[common_cols]
    else:
        Train_shapelet_filtered = pd.DataFrame(index=Train_shapelet.index)
        Test_shapelet_filtered = pd.DataFrame(index=Test_shapelet.index)
else:
    Train_shapelet_filtered = pd.DataFrame(index=Train_shapelet.index)
    Test_shapelet_filtered = pd.DataFrame(index=Test_shapelet.index)

# Merge shapelet features with existing features
X_train_combined = pd.concat(
    [X_train.reset_index(drop=True), Train_shapelet_filtered.reset_index(drop=True)],
    axis=1,
)
X_test_combined = pd.concat(
    [X_test.reset_index(drop=True), Test_shapelet_filtered.reset_index(drop=True)],
    axis=1,
)

# Re-apply categorical column settings
categorical_cols = [
    "race",
    "gender",
    "age_group",
    "insurance",
    "marital_status",
    "ethnicity",
]
for col in categorical_cols:
    if col in X_train_combined.columns:
        X_train_combined[col] = X_train_combined[col].astype("category")
        X_test_combined[col] = X_test_combined[col].astype("category")

# Save combined datasets
X_train_combined.to_csv(
    os.path.join(OUTPUT_DIR, "X_train_with_shapelets.csv"), index=False
)
X_test_combined.to_csv(
    os.path.join(OUTPUT_DIR, "X_test_with_shapelets.csv"), index=False
)

train_df_combined = pd.concat(
    [X_train_combined, y_train.reset_index(drop=True)], axis=1
)
test_df_combined = pd.concat([X_test_combined, y_test.reset_index(drop=True)], axis=1)
train_df_combined.to_csv(
    os.path.join(OUTPUT_DIR, "processed_train_with_shapelets.csv"), index=False
)
test_df_combined.to_csv(
    os.path.join(OUTPUT_DIR, "processed_test_with_shapelets.csv"), index=False
)
