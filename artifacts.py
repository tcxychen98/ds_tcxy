# on average, how long does it take to get the artifact with perfect main stats and perfect rolled sub stats?
# assuming you are grinding in the same domain
# all calculations are based on 5 star artifacts

import random
import numpy as np

art_type = input("Please enter your artifact type (fullname): ").upper()
if art_type == "FLOWER" or art_type == "FEATHER":
    pass
else:
    mainstat = input("What you want as your mainstat: ").upper()
substat1 = input("First substat: ").upper()
substat2 = input("Second substat: ").upper()
substat3 = input("Third substat: ").upper()


def flower_stats(sub1, sub2, sub3):
    flower_count = 1
    while True:
        sub_stats = ["ATK", "DEF", "HP%", "ATK%", "DEF%", "ER", "EM", "CR", "CD"]
        dist1 = [0.1578, 0.1579, 0.1053, 0.1053, 0.1053, 0.1053, 0.1053, 0.0789, 0.0789]
        obtained_stats = []
        sublist = [sub1, sub2, sub3]
        for i in sublist:
            if i == "RAND":
                sublist.remove(i)
        substat = np.random.choice(sub_stats, size=3, replace=False, p=dist1)
        for i in range(len(substat)):
            obtained_stats.append(str(substat[i]))
        flower_count += 1
        check = any(item in obtained_stats for item in sublist)
        if check is True:
            break
    return [obtained_stats, flower_count]

def feather_stats(sub1, sub2, sub3):
    feather_count = 1
    while True:
        sub_stats = ["HP", "DEF", "HP%", "ATK%", "DEF%", "ER", "EM", "CR", "CD"]
        dist1 = [0.1578, 0.1579, 0.1053, 0.1053, 0.1053, 0.1053, 0.1053, 0.0789, 0.0789]
        obtained_stats = []
        substat = np.random.choice(sub_stats, size=3, replace=False, p=dist1)
        sublist = [sub1, sub2, sub3]
        for i in range(len(substat)):
            obtained_stats.append(str(substat[i]))
        feather_count += 1
        check = all(item in obtained_stats for item in sublist)
        if check is True:
            break
    return [obtained_stats, feather_count]

def sands_stats(main1, sub1, sub2, sub3):
    sands_count = 1
    while True:
        main_stats = ["HP%", "ATK%", "DEF%", "ER", "EM"]
        dist1 = [0.2668, 0.2666, 0.2666, 0.1, 0.1]
        sub_stats = ["HP", "ATK", "DEF", "HP%", "ATK%", "DEF%", "ER", "EM", "CR", "CD"]
        dist2 = [0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.075, 0.075]
        obtained_stats = []
        mainstat = random.choices(main_stats, dist1)
        obtained_stats.append(mainstat[0])
        sub_stats.remove(mainstat[0])
        substat = np.random.choice(sub_stats, size=3, replace=False, p=dist2)
        sublist = [main1, sub1, sub2, sub3]
        for i in range(len(substat)):
            obtained_stats.append(str(substat[i]))
        sands_count += 1
        check = all(item in obtained_stats for item in sublist)
        if check is True:
            break
    return [obtained_stats, sands_count]

def goblet_stats(main1, sub1, sub2, sub3):
    goblet_count = 1
    while True:
        main_stats = ["HP%", "ATK%", "DEF%", "PYRO", "ELECTRO", "CRYO", "HYDRO", "ANEMO", "GEO", "PHYS", "EM"]
        dist1 = [0.2125, 0.2125, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.025]
        obtained_stats = []
        mainstat = random.choices(main_stats, dist1)
        obtained_stats.append(mainstat[0])
        if (mainstat[0] == "HP%") or (mainstat[0] == "ATK%") or (mainstat[0] == "DEF%") or (mainstat[0] == "EM"):
            sub_stats = ["HP", "ATK", "DEF", "HP%", "ATK%", "DEF%", "ER", "EM", "CR", "CD"]
            dist2a = [0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.075, 0.075]
            sub_stats.remove(mainstat[0])
            substat = np.random.choice(sub_stats, size=3, replace=False, p=dist2a)
            sublist = [main1, sub1, sub2, sub3]
            for i in range(len(substat)):
                obtained_stats.append(str(substat[i]))
        else:
            sub_stats = ["HP", "ATK", "DEF", "HP%", "ATK%", "DEF%", "ER", "EM", "CR", "CD"]
            dist2b = [0.1363, 0.1364, 0.1364, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0682, 0.0682]
            substat = np.random.choice(sub_stats, size=3, replace=False, p=dist2b)
            sublist = [main1, sub1, sub2, sub3]
            for i in range(len(substat)):
                obtained_stats.append(str(substat[i]))
        goblet_count += 1
        check = all(item in obtained_stats for item in sublist)
        if check is True:
            break
    return [obtained_stats, goblet_count]

def circlet_stats(main1, sub1, sub2, sub3):
    circ_count = 1
    while True:
        main_stats = ["HP%", "ATK%", "DEF%", "CD", "CR", "Heal", "EM"]
        dist1 = [0.22, 0.22, 0.22, 0.1, 0.1, 0.1, 0.04]
        sub_stats = ["HP", "ATK", "DEF", "HP%", "ATK%", "DEF%", "ER", "EM", "CR", "CD"]
        obtained_stats = []
        mainstat = random.choices(main_stats, dist1)
        obtained_stats.append(mainstat[0])
        if (mainstat[0] == "HP%") or (mainstat[0] == "ATK%") or (mainstat[0] == "DEF%") or (mainstat[0] == "EM"):
            dist2a = [0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.075, 0.075]
            sub_stats.remove(mainstat[0])
            substat = np.random.choice(sub_stats, size=3, replace=False, p=dist2a)
            sublist = [main1, sub1, sub2, sub3]
            for i in range(len(substat)):
                obtained_stats.append(str(substat[i]))
        elif (mainstat[0] == "CR") or (mainstat[0] == "CD"):
            dist2b = [0.1462, 0.1463, 0.1463, 0.0976, 0.0976, 0.0976, 0.0976, 0.0976, 0.0732]
            sub_stats.remove(mainstat[0])
            substat = np.random.choice(sub_stats, size=3, replace=False, p=dist2b)
            sublist = [main1, sub1, sub2, sub3]
            for i in range(len(substat)):
                obtained_stats.append(str(substat[i]))
        else:
            dist2c = [0.1363, 0.1364, 0.1364, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0682, 0.0682]
            substat = np.random.choice(sub_stats, size=3, replace=False, p=dist2c)
            sublist = [main1, sub1, sub2, sub3]
            for i in range(len(substat)):
                obtained_stats.append(str(substat[i]))
        circ_count += 1
        if obtained_stats[0] == main1:
            check = all(item in obtained_stats for item in sublist)
            if check is True:
                break
    return [obtained_stats, circ_count]

print(flower_stats(substat1, substat2, substat3))

# if mainstats == "SKIP":
#     if art_type == "FLOWER":
#         print(flower_stats(substat1,substat2,substat3))
#     else:
#         print(feather_stats(substat1,substat2,substat3))
# else:
#     if art_type == "SANDS":
#         print(sands_stats(mainstats,substat1,substat2,substat3))
#     elif art_type == "GOBLET":
#         print(goblet_stats(mainstats,substat1,substat2,substat3))
#     else:
#         print(circlet_stats(mainstats,substat1,substat2,substat3))