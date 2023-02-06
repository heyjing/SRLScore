from typing import List
import os

from bs4 import BeautifulSoup
from tqdm import tqdm


def create_verb_loopup(file_names: List[str]):
    verbs_info = []

    for file_name in tqdm(file_names):
        file_path = "../propbank-frames/frames/" + file_name
        with open(file_path, "r") as f:
            data = f.read()
        Bs_data = BeautifulSoup(data, "xml")
        Bs_roles = Bs_data.find_all("role", {"n": ["2", "3", "4"]})

        roles = set()
        verb_dic = {
            "verb": file_name[: file_name.index(".")],
            "roles": roles,
        }

        for element in Bs_roles:

            if element["descr"] == "recipient" or element["descr"] == "benefactive":
                roles.add("ARG" + element["n"])

            rol = element.find_all("rolelink", text="recipient" or "benefactive")
            if len(rol) != 0:
                roles.add("ARG" + element["n"])

        if len(roles) != 0:
            verbs_info.append(verb_dic)
    return verbs_info


if __name__ == "__main__":
    all_files_and_folders = os.listdir("../propbank-frames/frames/")
    all_files_and_folders.remove(".gitignore")
    print(len(all_files_and_folders))  # 7567 verbs

    # verbs_info = create_verb_loopup(["open.xml", "send.xml"])
    verbs_info = create_verb_loopup(all_files_and_folders)

    # with open("verbs_recipient_info.txt", "w") as fp:
    #     for item in verbs_info:
    #         # write each item on a new line
    #         fp.write("%s\n" % item)
    #     print("Done")
