import zipfile
import shutil
import json
import os

with open('kaggle.json') as f:
    account = json.load(f)
    os.environ['KAGGLE_USERNAME'] = account["username"]
    os.environ['KAGGLE_KEY'] = account["key"]
import kaggle


def clear_directory(path_to_dir: str, force: bool):
    if force and os.path.exists(path_to_dir):
        shutil.rmtree(path_to_dir)
    if force or not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)
        return True
    if not os.listdir(path_to_dir):
        return True

    return False


def download_dataset(kaggle_dataset_id: str, is_competition_rel: bool, out_directory: str, overwrite: bool = False):
    """
    :param kaggle_dataset_id: уникальное название набора данных для скачивания
    :param is_competition_rel: связан ли набор данных с соревнованием или он выложен отдельно
    :param out_directory: директория, в которую будет скачан набор данных
    :param overwrite: перезаписать ли предыдущие данные в директории
    """
    if not clear_directory(out_directory, overwrite):
        print("Directory is already occupied")
        return

    if is_competition_rel:
        kaggle.api.competition_download_files(kaggle_dataset_id,
                                              path=out_directory,
                                              quiet=False)
    else:
        kaggle.api.dataset_download_files(kaggle_dataset_id,
                                          path=out_directory,
                                          quiet=False)
    filename = f"{out_directory}/{kaggle_dataset_id.split('/')[-1]}.zip"
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(out_directory)
    os.remove(filename)
