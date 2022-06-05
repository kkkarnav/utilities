import os


def filter_dir_by_extension(directory, extension):
    filtered_files = [
        file
        for file in os.listdir(directory)
        if file.split(".")[-1] == extension.strip(".")
    ]
    return filtered_files


def filter_file_by_extension(file, extension):
    extension_in_file = file.split(".")[-1] == extension.strip(".")
    return extension_in_file


def delete_files(directory, files):
    [os.remove(directory + "\\" + file) for file in files]


if __name__ == "__main__":
    directory = "D:\\code\\scripts\\ceda"

    delete_files(directory, filter_dir_by_extension(directory, ".csv"))
