# # # import numpy as np
# # # import torch
# # # import pandas as pd
# # # import argparse
# # # import torch.nn as nn
# # # import pickle


# # # class Generator(nn.Module):
# # #     def __init__(self, input_dim, latent_dim=100):
# # #         super(Generator, self).__init__()
# # #         self.model = nn.Sequential(
# # #             nn.Linear(latent_dim, 256),
# # #             nn.ReLU(True),
# # #             nn.BatchNorm1d(256),
# # #             nn.Linear(256, 512),
# # #             nn.ReLU(True),
# # #             nn.BatchNorm1d(512),
# # #             nn.Linear(512, 1024),
# # #             nn.ReLU(True),
# # #             nn.BatchNorm1d(1024),
# # #             nn.Linear(1024, input_dim),
# # #             nn.Tanh()
# # #         )

# # #     def forward(self, z):
# # #         return self.model(z)


# # # def augment_data(X_train_minority, X_train_majority, y_train_minority, y_train_majority, generator, latent_dim=100):
# # #     num_samples_to_generate = len(X_train_minority)
# # #     noise = torch.randn(num_samples_to_generate, latent_dim)
# # #     generated_samples = generator(noise).detach().numpy()

# # #     X_augmented = np.vstack((X_train_minority, generated_samples))
# # #     y_augmented = np.hstack(
# # #         (y_train_minority, [y_train_minority.iloc[0]] * num_samples_to_generate))

# # #     X_final = np.vstack((X_augmented, X_train_majority))
# # #     y_final = np.hstack((y_augmented, y_train_majority))

# # #     return X_final, y_final


# # # if __name__ == "__main__":
# # #     parser = argparse.ArgumentParser(description="Augment Data")
# # #     parser.add_argument('--input-dir', type=str, required=True)
# # #     parser.add_argument('--output', type=str, required=True)
# # #     parser.add_argument('--generator-model', type=str, required=True)
# # #     args = parser.parse_args()

# # #     X_train = pd.read_csv(f"{args.input_dir}/X_train.csv")
# # #     y_train = pd.read_csv(f"{args.input_dir}/y_train.csv")

# # #     # Load the entire generator object
# # #     with open(args.generator_model, 'rb') as f:
# # #         generator = pickle.load(f)

# # #     minority_class = y_train['Class'].value_counts().idxmin()
# # #     majority_class = y_train['Class'].value_counts().idxmax()

# # #     X_train_minority = X_train[y_train['Class'] == minority_class]
# # #     X_train_majority = X_train[y_train['Class'] == majority_class]
# # #     y_train_minority = y_train[y_train['Class'] == minority_class]
# # #     y_train_majority = y_train[y_train['Class'] == majority_class]

# # #     X_final, y_final = augment_data(
# # #         X_train_minority, X_train_majority, y_train_minority, y_train_majority, generator)

# # #     X_final_df = pd.DataFrame(X_final, columns=X_train.columns)
# # #     y_final_df = pd.DataFrame(y_final, columns=y_train.columns)

# # #     X_final_df.to_csv(f"{args.output}/X_augmented.csv", index=False)
# # #     y_final_df.to_csv(f"{args.output}/y_augmented.csv", index=False)

# # import numpy as np
# # import torch
# # import pandas as pd
# # import argparse
# # import torch.nn as nn
# # import torch.optim as optim
# # import argparse
# # import pickle


# # class Generator(nn.Module):
# #     def __init__(self, input_dim, latent_dim=100):
# #         super(Generator, self).__init__()
# #         self.model = nn.Sequential(
# #             nn.Linear(latent_dim, 256),
# #             nn.ReLU(True),
# #             nn.BatchNorm1d(256),
# #             nn.Linear(256, 512),
# #             nn.ReLU(True),
# #             nn.BatchNorm1d(512),
# #             nn.Linear(512, 1024),
# #             nn.ReLU(True),
# #             nn.BatchNorm1d(1024),
# #             nn.Linear(1024, input_dim),
# #             nn.Tanh()
# #         )

# #     def forward(self, z):
# #         return self.model(z)


# # def augment_data(X_train_minority, X_train_majority, y_train_minority, y_train_majority, generator, latent_dim=100):
# #     num_samples_to_generate = len(X_train_minority)
# #     noise = torch.randn(num_samples_to_generate, latent_dim)
# #     generated_samples = generator(noise).detach().numpy()

# #     X_augmented = np.vstack((X_train_minority, generated_samples))
# #     y_augmented = np.hstack(
# #         (y_train_minority, [y_train_minority.iloc[1]] * num_samples_to_generate))
# #     print(y_augmented.shape)
# #     print(y_train_majority)
# #     X_final = np.vstack((X_augmented, X_train_majority))
# #     y_final = np.hstack((y_augmented, y_train_majority))

# #     return X_final, y_final


# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser(description="Augment Data")
# #     parser.add_argument('--input-dir', type=str, required=True)
# #     parser.add_argument('--output', type=str, required=True)
# #     parser.add_argument('--generator-model', type=str, required=True)
# #     args = parser.parse_args()

# #     X_train = pd.read_csv(f"{args.input_dir}/X_train.csv")
# #     y_train = pd.read_csv(f"{args.input_dir}/y_train.csv")

# #     # generator = Generator(input_dim=X_train.shape[1])
# #     # generator.load_state_dict(torch.load("generator.pth"))
# #     with open('models/generator.pkl', 'rb') as f:
# #         generator = pickle.load(f)

# #     minority_class = y_train['Class'].value_counts().idxmin()

# #     X_train_minority = X_train[y_train['Class'] == minority_class]
# #     X_train_majority = X_train[y_train['Class'] != minority_class]
# #     y_train_minority = y_train[y_train['Class'] == minority_class]
# #     y_train_majority = y_train[y_train['Class'] != minority_class]

# #     X_final, y_final = augment_data(
# #         X_train_minority, X_train_majority, y_train_minority, y_train_majority, generator)

# #     X_final_df = pd.DataFrame(X_final, columns=X_train.columns)
# #     y_final_df = pd.DataFrame(y_final, columns=y_train.columns)

# #     X_final_df.to_csv(f"{args.output}/X_augmented.csv", index=False)
# #     y_final_df.to_csv(f"{args.output}/y_augmented.csv", index=False)
# import numpy as np
# import torch
# import pandas as pd
# import argparse
# import torch.nn as nn
# import torch.optim as optim
# import argparse
# import pickle


# class Generator(nn.Module):
#     def __init__(self, input_dim, latent_dim=100):
#         super(Generator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(latent_dim, 256),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.Linear(256, 512),
#             nn.ReLU(True),
#             nn.BatchNorm1d(512),
#             nn.Linear(512, 1024),
#             nn.ReLU(True),
#             nn.BatchNorm1d(1024),
#             nn.Linear(1024, input_dim),
#             nn.Tanh()
#         )

#     def forward(self, z):
#         return self.model(z)


# def augment_data(X_train_minority, X_train_majority, y_train_minority, y_train_majority, generator, latent_dim=100):
#     num_samples_to_generate = len(X_train_minority)
#     noise = torch.randn(num_samples_to_generate, latent_dim)
#     generated_samples = generator(noise).detach().numpy()

#     # Print shapes for observation
#     print("Shapes before augmentation:")
#     print("X_train_minority shape:", X_train_minority.shape)
#     print("X_train_majority shape:", X_train_majority.shape)
#     print("y_train_minority shape:", y_train_minority.shape)
#     print("y_train_majority shape:", y_train_majority.shape)
#     print("Generated samples shape:", generated_samples.shape)

#     X_augmented = np.vstack((X_train_minority, generated_samples))
#     y_augmented = np.hstack(
#         # Corrected line
#         (y_train_minority, np.array([y_train_minority.iloc[0]] * num_samples_to_generate)))

#     # Print shapes after augmenting minority class
#     print("\nShapes after augmenting minority class:")
#     print("X_augmented shape:", X_augmented.shape)
#     print("y_augmented shape:", y_augmented.shape)

#     # Ensure X_train_majority and y_train_majority have compatible shapes
#     assert X_augmented.shape[1] == X_train_majority.shape[1], \
#         "Dimensions of X_augmented and X_train_majority do not match along axis 1."

#     X_final = np.vstack((X_augmented, X_train_majority))
#     y_final = np.hstack((y_augmented, y_train_majority))

#     # Print final shapes after combining both classes
#     print("\nFinal shapes after combining both classes:")
#     print("X_final shape:", X_final.shape)
#     print("y_final shape:", y_final.shape)

#     return X_final, y_final


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Augment Data")
#     parser.add_argument('--input-dir', type=str, required=True)
#     parser.add_argument('--output', type=str, required=True)
#     parser.add_argument('--generator-model', type=str, required=True)
#     args = parser.parse_args()

#     X_train = pd.read_csv(f"{args.input_dir}/X_train.csv")
#     y_train = pd.read_csv(f"{args.input_dir}/y_train.csv")

#     # Load generator model
#     with open(args.generator_model, 'rb') as f:
#         generator = pickle.load(f)

#     minority_class = y_train['Class'].value_counts().idxmin()

#     X_train_minority = X_train[y_train['Class'] == minority_class]
#     X_train_majority = X_train[y_train['Class'] != minority_class]
#     y_train_minority = y_train[y_train['Class'] == minority_class]
#     y_train_majority = y_train[y_train['Class'] != minority_class]

#     X_final, y_final = augment_data(
#         X_train_minority, X_train_majority, y_train_minority, y_train_majority, generator)

#     X_final_df = pd.DataFrame(X_final, columns=X_train.columns)
#     y_final_df = pd.DataFrame(y_final, columns=y_train.columns)

#     X_final_df.to_csv(f"{args.output}/X_augmented.csv", index=False)
#     y_final_df.to_csv(f"{args.output}/y_augmented.csv", index=False)
import numpy as np
import torch
import pandas as pd
import argparse
import torch.nn as nn
import torch.optim as optim
import argparse
import pickle


class Generator(nn.Module):
    def __init__(self, input_dim, latent_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, input_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


def augment_data(X_train_minority, X_train_majority, y_train_minority, y_train_majority, generator, latent_dim=100):
    num_samples_to_generate = len(X_train_minority)
    noise = torch.randn(num_samples_to_generate, latent_dim)
    generated_samples = generator(noise).detach().numpy()

    # Print shapes for observation
    print("Shapes before augmentation:")
    print("X_train_minority shape:", X_train_minority.shape)
    print("X_train_majority shape:", X_train_majority.shape)
    print("y_train_minority shape:", y_train_minority.shape)
    print("y_train_majority shape:", y_train_majority.shape)
    print("Generated samples shape:", generated_samples.shape)

    X_augmented = np.vstack((X_train_minority, generated_samples))

    # Create an array of minority class labels matching the number of generated samples
    minority_label = y_train_minority.iloc[0].values
    y_augmented = np.vstack((y_train_minority.values, np.tile(
        minority_label, (num_samples_to_generate, 1))))

    # Print shapes after augmenting minority class
    print("\nShapes after augmenting minority class:")
    print("X_augmented shape:", X_augmented.shape)
    print("y_augmented shape:", y_augmented.shape)

    # Ensure X_train_majority and y_train_majority have compatible shapes
    assert X_augmented.shape[1] == X_train_majority.shape[1], \
        "Dimensions of X_augmented and X_train_majority do not match along axis 1."

    X_final = np.vstack((X_augmented, X_train_majority))
    y_final = np.vstack((y_augmented, y_train_majority))

    # Print final shapes after combining both classes
    print("\nFinal shapes after combining both classes:")
    print("X_final shape:", X_final.shape)
    print("y_final shape:", y_final.shape)

    return X_final, y_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment Data")
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--generator-model', type=str, required=True)
    args = parser.parse_args()

    X_train = pd.read_csv(f"{args.input_dir}/X_train.csv")
    y_train = pd.read_csv(f"{args.input_dir}/y_train.csv")

    # Load generator model
    with open(args.generator_model, 'rb') as f:
        generator = pickle.load(f)

    minority_class = y_train['Class'].value_counts().idxmin()

    X_train_minority = X_train[y_train['Class'] == minority_class]
    X_train_majority = X_train[y_train['Class'] != minority_class]
    y_train_minority = y_train[y_train['Class'] == minority_class]
    y_train_majority = y_train[y_train['Class'] != minority_class]

    X_final, y_final = augment_data(
        X_train_minority, X_train_majority, y_train_minority, y_train_majority, generator)

    X_final_df = pd.DataFrame(X_final, columns=X_train.columns)
    y_final_df = pd.DataFrame(y_final, columns=y_train.columns)

    X_final_df.to_csv(f"{args.output}/X_augmented.csv", index=False)
    y_final_df.to_csv(f"{args.output}/y_augmented.csv", index=False)
