import json

# 读取现有的JSON文件
with open('../test_json/long_test_two.json', 'r') as file:
    data = json.load(file)

# 添加新的项
for i in range(5, 40):  # 这里假设要添加从 l5 到 l9 的项
    key = f"l{i}"
    data[key] = {
        "video_path": f"../cogvideo_test_sample/{i}.mp4",
        "prompt": "Two girls are talking happily.",
        "reference_image":0
    }

# 将更新后的数据写回JSON文件
with open('../test_json/long_test_two.json', 'w') as file:
    json.dump(data, file, indent=4)

print("新的项已成功添加到JSON文件中。")
