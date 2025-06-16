ANIMAL_CATEGORIES = {
    "家犬 (Domestic Dog)": list(range(151, 269)),
    "猫科动物 (Cat)": list(range(281, 294)),  # 合并了家猫和野猫
    "宠物鸟 (Pet Bird)": [88, 89, 90],
    "小型宠物 (Small Pet)": [333, 337],
    "鸟类 (Bird)": list(range(7, 147)),
    "爬行与两栖动物 (Reptile/Amphibian)": list(range(25, 27)) + list(range(30, 38)) + list(range(48, 73)),
    "鱼类 (Fish)": list(range(0, 3)) + list(range(389, 398)),
    "昆虫与蛛形纲 (Insect/Arachnid)": list(range(73, 77)) + list(range(300, 321)),
    "野生哺乳动物 (Wild Mammal)": list(range(269, 281)) + list(range(294, 389)),  # 包含了野生犬科和其他哺乳动物
}

def classify_animal_from_index(pred_index):
    if pred_index in ANIMAL_CATEGORIES["家犬 (Domestic Dog)"]:
        return "家犬 (Domestic Dog)"
    if pred_index in ANIMAL_CATEGORIES["猫科动物 (Cat)"]:
        if 281 <= pred_index <= 285:
            return "家猫 (Domestic Cat)"
        else:
            return "野生猫科 (Wild Cat)"
    if pred_index in ANIMAL_CATEGORIES["宠物鸟 (Pet Bird)"]:
        return "宠物鸟 (Pet Bird)"
    if pred_index in ANIMAL_CATEGORIES["小型宠物 (Small Pet)"]:
        return "小型宠物 (Small Pet)"

    for category, index_list in ANIMAL_CATEGORIES.items():
        if pred_index in index_list:
            return category

    return "非动物 (Not an Animal)"